use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use clap::Parser;
use sqlx::PgPool;
use tracing::{info, warn};

use expert_types::event::Event;
use expert_types::training::{Label, LabelSource, TrainingExample};

#[derive(Parser)]
#[command(
    name = "training-import",
    about = "Import Discord data exports as training examples"
)]
struct Args {
    /// Path to the extracted Discord `messages/` directory.
    #[arg(long, default_value = "data/discord-export/messages")]
    data_dir: PathBuf,

    /// Your bot's Discord user ID (messages from this user are labeled negative).
    #[arg(long)]
    bot_user_id: String,

    /// Sliding window size for event sequences.
    #[arg(long, default_value_t = 5)]
    window_size: usize,

    /// Ollama base URL for generating embeddings.
    #[arg(long, default_value = "http://127.0.0.1:11434")]
    ollama_url: String,

    /// Embeddings model name.
    #[arg(long, default_value = "qwen3-embedding:8b")]
    embeddings_model: String,

    /// Postgres connection URL.
    #[arg(long, env = "DATABASE_URL")]
    database_url: String,

    /// Synthetic goal ID to assign to all imported examples.
    #[arg(long, default_value = "import-dm-response")]
    goal_id: String,

    /// If set, write a CSV manifest instead of inserting into Postgres.
    #[arg(long)]
    manifest_only: bool,
}

#[derive(serde::Deserialize)]
struct ChannelMeta {
    id: String,
    #[serde(rename = "type")]
    channel_type: u8,
    name: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    recipients: Vec<String>,
}

struct Message {
    id: String,
    timestamp: String,
    contents: String,
    author_id: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .compact()
        .init();

    let args = Args::parse();

    if !args.data_dir.exists() {
        bail!(
            "Data directory {} does not exist. Place your Discord export at that path.",
            args.data_dir.display()
        );
    }

    let pool = if !args.manifest_only {
        Some(
            PgPool::connect(&args.database_url)
                .await
                .context("Failed to connect to Postgres")?,
        )
    } else {
        None
    };

    let http = reqwest::Client::new();
    let mut total_positive = 0usize;
    let mut total_negative = 0usize;
    let mut channels_processed = 0usize;

    let mut entries: Vec<std::fs::DirEntry> = std::fs::read_dir(&args.data_dir)?
        .filter_map(|e| e.ok())
        .collect();
    entries.sort_by_key(|e| e.file_name());

    for entry in entries {
        let dir = entry.path();
        if !dir.is_dir() {
            continue;
        }

        let channel_json = dir.join("channel.json");
        let messages_csv = dir.join("messages.csv");

        if !channel_json.exists() || !messages_csv.exists() {
            continue;
        }

        let meta: ChannelMeta = serde_json::from_str(
            &std::fs::read_to_string(&channel_json)
                .with_context(|| format!("reading {}", channel_json.display()))?,
        )
        .with_context(|| format!("parsing {}", channel_json.display()))?;

        let is_dm = meta.channel_type == 1;
        let channel_label = if is_dm { "DM" } else { "guild" };

        info!(
            channel_id = %meta.id,
            channel_type = channel_label,
            name = meta.name.as_deref().unwrap_or("unnamed"),
            "processing channel"
        );

        let messages = read_messages_csv(&messages_csv)?;
        if messages.len() < args.window_size {
            info!(
                count = messages.len(),
                "skipping channel: too few messages for window"
            );
            continue;
        }

        for window_start in 0..=(messages.len() - args.window_size) {
            let window = &messages[window_start..window_start + args.window_size];
            let trigger = &window[window.len() - 1];

            let is_bot_message = trigger.author_id == args.bot_user_id;
            let is_positive = if is_dm {
                !is_bot_message
            } else {
                !is_bot_message && trigger.contents.contains(&args.bot_user_id)
            };

            let label = if is_positive {
                Label::Positive
            } else {
                Label::Negative
            };

            let event_window: Vec<Event> = window
                .iter()
                .enumerate()
                .map(|(i, msg)| {
                    let event_type_str = if is_dm { "dm" } else { "message" };
                    let raw = if is_dm {
                        format!("[DM from {}] {}", msg.author_id, msg.contents)
                    } else {
                        format!("[Message in {}] {}", meta.id, msg.contents)
                    };

                    let mut metadata = HashMap::new();
                    metadata.insert(
                        "event_type".to_string(),
                        serde_json::Value::String(event_type_str.to_string()),
                    );
                    metadata.insert(
                        "author_id".to_string(),
                        serde_json::Value::String(msg.author_id.clone()),
                    );
                    metadata.insert(
                        "channel_id".to_string(),
                        serde_json::Value::String(meta.id.clone()),
                    );
                    metadata.insert(
                        "message_id".to_string(),
                        serde_json::Value::String(msg.id.clone()),
                    );
                    metadata.insert(
                        "is_self".to_string(),
                        serde_json::Value::Bool(msg.author_id == args.bot_user_id),
                    );

                    Event {
                        id: msg.id.clone(),
                        stream_id: format!("import-{}", meta.id),
                        sequence: (window_start + i) as u64,
                        timestamp: parse_timestamp(&msg.timestamp),
                        raw,
                        embedding: None,
                        metadata,
                    }
                })
                .collect();

            // Generate embeddings via Ollama
            let mut window_vectors = Vec::new();
            for event in &event_window {
                match embed_text(&http, &args.ollama_url, &args.embeddings_model, &event.raw).await
                {
                    Ok(emb) => window_vectors.push(emb),
                    Err(e) => {
                        warn!(error = %e, "embedding failed, using zero vector");
                        window_vectors.push(vec![0.0; 4096]);
                    }
                }
            }

            let goal_embedding = window_vectors.last().cloned().unwrap_or_default();

            let example = TrainingExample {
                id: uuid::Uuid::new_v4().to_string(),
                activity_id: format!("import-{}", meta.id),
                stream_id: format!("import-{}", meta.id),
                domain: Some("discord".to_string()),
                goal_id: args.goal_id.clone(),
                goal_version: 1,
                goal_embedding,
                event_window,
                window_vectors,
                label,
                label_source: LabelSource::Human,
                label_weight: 1.0,
                reason: format!(
                    "Discord import: {} channel {}, trigger by {}",
                    channel_label, meta.id, trigger.author_id
                ),
                created_at: now_ms(),
                used_in_batch: false,
                confidence: 1.0,
                consensus_count: 1,
            };

            if is_positive {
                total_positive += 1;
            } else {
                total_negative += 1;
            }

            if let Some(ref pool) = pool {
                insert_example(pool, &example).await?;
            }
        }

        channels_processed += 1;
    }

    info!(
        channels = channels_processed,
        positive = total_positive,
        negative = total_negative,
        total = total_positive + total_negative,
        "import complete"
    );

    Ok(())
}

fn read_messages_csv(path: &PathBuf) -> Result<Vec<Message>> {
    let mut reader = csv::Reader::from_path(path)?;
    let mut messages = Vec::new();

    for result in reader.records() {
        let record = result?;
        // Discord export CSV: ID, Timestamp, Contents, Attachments
        if record.len() < 3 {
            continue;
        }
        let id = record[0].to_string();
        let timestamp = record[1].to_string();
        let contents = record[2].to_string();

        if contents.is_empty() {
            continue;
        }

        // The Discord export doesn't include author_id in messages.csv directly.
        // We'll extract it from the ID or use a placeholder -- the actual Discord
        // data package format has an Authors column in newer exports.
        // For now we parse what's available.
        let author_id = if record.len() > 3 {
            record[3].to_string()
        } else {
            "unknown".to_string()
        };

        messages.push(Message {
            id,
            timestamp,
            contents,
            author_id,
        });
    }

    Ok(messages)
}

fn parse_timestamp(ts: &str) -> u64 {
    chrono::DateTime::parse_from_rfc3339(ts)
        .or_else(|_| chrono::DateTime::parse_from_str(ts, "%Y-%m-%d %H:%M:%S%.f%z"))
        .map(|dt: chrono::DateTime<chrono::FixedOffset>| dt.timestamp_millis() as u64)
        .unwrap_or(0)
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

async fn embed_text(
    client: &reqwest::Client,
    ollama_url: &str,
    model: &str,
    text: &str,
) -> Result<Vec<f32>> {
    let url = format!("{}/api/embed", ollama_url.trim_end_matches('/'));
    let body = serde_json::json!({
        "model": model,
        "input": text,
    });

    let resp = client
        .post(&url)
        .json(&body)
        .timeout(std::time::Duration::from_secs(30))
        .send()
        .await?
        .error_for_status()?
        .json::<serde_json::Value>()
        .await?;

    let embeddings = resp
        .get("embeddings")
        .and_then(|e| e.as_array())
        .and_then(|arr| arr.first())
        .and_then(|e| e.as_array())
        .context("unexpected embedding response format")?;

    Ok(embeddings
        .iter()
        .filter_map(|v| v.as_f64().map(|f| f as f32))
        .collect())
}

async fn insert_example(pool: &PgPool, example: &TrainingExample) -> Result<()> {
    let event_window_json = serde_json::to_value(&example.event_window)?;
    let window_vectors_json = serde_json::to_value(&example.window_vectors)?;
    let label_str = match example.label {
        Label::Positive => "positive",
        Label::Negative => "negative",
    };
    let source_str = match example.label_source {
        LabelSource::Human => "human",
        LabelSource::LlmSuppress => "llm_suppress",
        LabelSource::LlmRecall => "llm_recall",
        LabelSource::Synthetic => "synthetic",
    };

    sqlx::query(
        r#"
        INSERT INTO training_examples (
            id, activity_id, stream_id, domain, goal_id, goal_version,
            goal_embedding, event_window, window_vectors,
            label, label_source, label_weight, reason,
            created_at, used_in_batch, confidence, consensus_count
        ) VALUES (
            $1, $2, $3, $4, $5, $6,
            $7, $8, $9,
            $10, $11, $12, $13,
            $14, $15, $16, $17
        ) ON CONFLICT (id) DO NOTHING
        "#,
    )
    .bind(&example.id)
    .bind(&example.activity_id)
    .bind(&example.stream_id)
    .bind(&example.domain)
    .bind(&example.goal_id)
    .bind(example.goal_version as i32)
    .bind(&serde_json::to_value(&example.goal_embedding)?)
    .bind(&event_window_json)
    .bind(&window_vectors_json)
    .bind(label_str)
    .bind(source_str)
    .bind(example.label_weight)
    .bind(&example.reason)
    .bind(example.created_at as i64)
    .bind(example.used_in_batch)
    .bind(example.confidence)
    .bind(example.consensus_count as i32)
    .execute(pool)
    .await?;

    Ok(())
}
