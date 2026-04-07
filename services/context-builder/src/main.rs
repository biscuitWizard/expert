use std::time::Duration;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tracing::{error, info, warn};

use expert_config::Config;
use expert_redis::names;
use expert_redis::{ServiceLogger, StreamConsumer, StreamProducer};
use expert_types::context::{ContextPackage, Episode, Exchange, SelfKnowledgeNode};
use expert_types::event::Event;
use expert_types::signals::AssembleRequest;

#[derive(Serialize)]
struct RagQuery {
    request_id: String,
    query_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    embedding: Option<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    activity_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    k: Option<usize>,
}

#[derive(Deserialize)]
struct RagResult {
    request_id: String,
    #[serde(default)]
    episodes: Vec<Episode>,
    #[serde(default)]
    exchanges: Vec<Exchange>,
    #[serde(default)]
    compressed_history: Option<String>,
    #[serde(default)]
    self_knowledge: Vec<SelfKnowledgeNode>,
}

#[tokio::main]
async fn main() -> Result<()> {
    expert_config::init_tracing();
    let config = Config::from_env();
    info!("starting context-builder");

    let conn = expert_redis::connect(&config.redis_url).await?;
    let mut producer = StreamProducer::new(conn.clone(), config.stream_maxlen);

    let mut consumer = StreamConsumer::new(
        conn.clone(),
        names::REQUESTS_CONTEXT.to_string(),
        "ctx".to_string(),
        "ctx-0".to_string(),
        500,
    )
    .await?;

    let mut rag_producer = StreamProducer::new(conn.clone(), config.stream_maxlen);
    let mut svc_log = ServiceLogger::new(producer.clone(), "context-builder");

    loop {
        match consumer.consume::<AssembleRequest>().await {
            Ok(Some((id, req))) => {
                let _ = consumer.ack(&id).await;
                info!(
                    activity_id = %req.activity_id,
                    stream_id = %req.stream_id,
                    "assembling context package"
                );

                match assemble(&req, &config, &mut conn.clone(), &mut rag_producer).await {
                    Ok(package) => {
                        if let Err(e) = producer.publish(names::PACKAGES_READY, &package).await {
                            error!(error = %e, "failed to publish context package");
                            svc_log
                                .error(
                                    format!(
                                        "Failed to publish context package for {}: {e}",
                                        &req.activity_id[..8]
                                    ),
                                    None,
                                )
                                .await;
                        } else {
                            info!(activity_id = %req.activity_id, "context package published");
                        }
                    }
                    Err(e) => {
                        error!(error = %e, activity_id = %req.activity_id, "context assembly failed");
                        svc_log
                            .error(
                                format!(
                                    "Context assembly failed for {}: {e}",
                                    &req.activity_id[..8]
                                ),
                                Some(serde_json::json!({ "activity_id": req.activity_id })),
                            )
                            .await;
                    }
                }
            }
            Ok(None) => {}
            Err(e) => {
                warn!(error = %e, "context consumer error");
                tokio::time::sleep(Duration::from_secs(1)).await;
            }
        }
    }
}

async fn assemble(
    req: &AssembleRequest,
    config: &Config,
    conn: &mut redis::aio::MultiplexedConnection,
    rag_producer: &mut StreamProducer,
) -> Result<ContextPackage> {
    // 1. XREVRANGE lookback for recent events
    let stream = names::events_embedded(&req.stream_id);
    let end = &req.fire_signal.trigger_event_seq;
    let start = req.fire_signal.last_fired_seq.as_deref().unwrap_or("0");

    let events: Vec<(String, Event)> = expert_redis::streams::xrevrange(
        conn,
        &stream,
        end,
        start,
        config.context_max_recent_events,
    )
    .await
    .unwrap_or_default();

    let mut recent_events: Vec<Event> = events.into_iter().map(|(_, e)| e).collect();
    recent_events.reverse(); // Chronological order

    // Identify trigger event (last in the list or by sequence match)
    let trigger_event = recent_events.last().cloned().unwrap_or_else(|| Event {
        id: String::new(),
        stream_id: req.stream_id.clone(),
        sequence: 0,
        timestamp: 0,
        raw: "[trigger event not found in lookback]".to_string(),
        embedding: None,
        metadata: Default::default(),
    });

    // 2. RAG query for relevant episodes (with timeout)
    let (retrieved_episodes, compressed_history, recent_exchanges, self_knowledge) =
        query_rag(req, config, &trigger_event, rag_producer, conn).await;

    // 3. Render natural language prompt
    let rendered_prompt = render_prompt(
        req,
        &trigger_event,
        &recent_events,
        &retrieved_episodes,
        &compressed_history,
        &recent_exchanges,
        &self_knowledge,
        config,
    );

    Ok(ContextPackage {
        activity_id: req.activity_id.clone(),
        firing_goals: req
            .fire_signal
            .firing_goal_ids
            .iter()
            .filter_map(|gid| req.goal_tree.iter().find(|g| &g.id == gid).cloned())
            .collect(),
        trigger_event,
        trigger_scores: req.fire_signal.scores.clone(),
        recent_events,
        retrieved_episodes,
        compressed_history,
        recent_raw_exchanges: recent_exchanges,
        rendered_prompt,
        tool_definitions: req.tool_definitions.clone(),
    })
}

async fn query_rag(
    req: &AssembleRequest,
    config: &Config,
    trigger_event: &Event,
    rag_producer: &mut StreamProducer,
    conn: &mut redis::aio::MultiplexedConnection,
) -> (
    Vec<Episode>,
    Option<String>,
    Vec<Exchange>,
    Vec<SelfKnowledgeNode>,
) {
    let timeout_dur = Duration::from_millis(config.context_rag_timeout_ms);

    // Semantic search for relevant episodes
    let search_id = uuid::Uuid::new_v4().to_string();
    let search_query = RagQuery {
        request_id: search_id.clone(),
        query_type: "semantic_search".to_string(),
        embedding: trigger_event.embedding.clone(),
        activity_id: Some(req.activity_id.clone()),
        k: Some(config.context_max_episodes),
    };
    let _ = rag_producer
        .publish(names::QUERIES_RAG, &search_query)
        .await;

    // History query
    let history_id = uuid::Uuid::new_v4().to_string();
    let history_query = RagQuery {
        request_id: history_id.clone(),
        query_type: "get_history".to_string(),
        embedding: None,
        activity_id: Some(req.activity_id.clone()),
        k: Some(config.context_max_exchanges),
    };
    let _ = rag_producer
        .publish(names::QUERIES_RAG, &history_query)
        .await;

    // Self-knowledge query
    let sk_id = uuid::Uuid::new_v4().to_string();
    let sk_query = RagQuery {
        request_id: sk_id.clone(),
        query_type: "get_self_knowledge".to_string(),
        embedding: trigger_event.embedding.clone(),
        activity_id: None,
        k: Some(3),
    };
    let _ = rag_producer.publish(names::QUERIES_RAG, &sk_query).await;

    // Wait for results with timeout
    let mut episodes = Vec::new();
    let mut compressed_history = None;
    let mut exchanges = Vec::new();
    let mut self_knowledge = Vec::new();

    let mut result_consumer = match StreamConsumer::new(
        conn.clone(),
        names::RESULTS_RAG.to_string(),
        format!("ctx-rag-{search_id}"),
        "ctx-rag-0".to_string(),
        200,
    )
    .await
    {
        Ok(c) => c,
        Err(_) => return (episodes, compressed_history, exchanges, self_knowledge),
    };

    let deadline = tokio::time::Instant::now() + timeout_dur;
    let mut found_search = false;
    let mut found_history = false;
    let mut found_sk = false;

    while (!found_search || !found_history || !found_sk) && tokio::time::Instant::now() < deadline {
        match tokio::time::timeout(
            deadline.saturating_duration_since(tokio::time::Instant::now()),
            result_consumer.consume::<RagResult>(),
        )
        .await
        {
            Ok(Ok(Some((id, result)))) => {
                let _ = result_consumer.ack(&id).await;
                if result.request_id == search_id {
                    episodes = result.episodes;
                    found_search = true;
                } else if result.request_id == history_id {
                    compressed_history = result.compressed_history;
                    exchanges = result.exchanges;
                    found_history = true;
                } else if result.request_id == sk_id {
                    self_knowledge = result.self_knowledge;
                    found_sk = true;
                }
            }
            _ => break,
        }
    }

    (episodes, compressed_history, exchanges, self_knowledge)
}

fn render_prompt(
    req: &AssembleRequest,
    trigger_event: &Event,
    recent_events: &[Event],
    episodes: &[Episode],
    compressed_history: &Option<String>,
    recent_exchanges: &[Exchange],
    self_knowledge: &[SelfKnowledgeNode],
    config: &Config,
) -> String {
    let mut prompt = String::with_capacity(4096);

    // === YOUR IDENTITY ===
    prompt.push_str("=== YOUR IDENTITY ===\n");
    if let Some(identity) = &req.bot_identity {
        prompt.push_str(&format!(
            "You are Zero (username={}, user_id={}).\n",
            identity.username, identity.user_id
        ));
    } else {
        prompt.push_str("You are Zero.\n");
    }
    prompt.push_str("Events where is_self=true are your own messages.\n");
    for node in self_knowledge {
        prompt.push_str(&format!("{}\n", node.content));
    }
    prompt.push('\n');

    // === ACTIVITY CONTEXT ===
    prompt.push_str("=== ACTIVITY CONTEXT ===\n");
    prompt.push_str(&format!(
        "You are monitoring a live event stream (stream: {}, activity: {}).\n",
        req.stream_id, req.activity_id
    ));
    prompt.push_str(
        "Your role is to respond when relevant events occur. You have tools to provide feedback on whether this invocation was useful.\n\n"
    );

    // === YOUR CURRENT GOALS ===
    prompt.push_str("=== YOUR CURRENT GOALS ===\n");
    for goal in &req.goal_tree {
        let firing = if req.fire_signal.firing_goal_ids.contains(&goal.id) {
            " [TRIGGERED]"
        } else {
            ""
        };
        prompt.push_str(&format!(
            "- {}: {}{}\n",
            goal.name, goal.description, firing
        ));
    }
    prompt.push('\n');

    // === TRIGGER EVENT ===
    prompt.push_str("=== TRIGGER EVENT ===\n");
    prompt.push_str(&format!("{}\n", trigger_event.raw));
    render_event_metadata(&mut prompt, trigger_event);
    prompt.push('\n');

    // === RECENT STREAM ACTIVITY ===
    prompt.push_str("=== RECENT STREAM ACTIVITY ===\n");
    let max_recent = config.context_max_recent_events.min(recent_events.len());
    let display_events = &recent_events[recent_events.len().saturating_sub(max_recent)..];
    for (i, event) in display_events.iter().enumerate() {
        prompt.push_str(&format!("[{}] {}\n", i, event.raw));
        render_event_metadata(&mut prompt, event);
    }
    prompt.push('\n');

    // === RELEVANT PAST CONTEXT ===
    if !episodes.is_empty() {
        prompt.push_str("=== RELEVANT PAST CONTEXT ===\n");
        for ep in episodes.iter().take(config.context_max_episodes) {
            prompt.push_str(&format!("Past episode: {}\n", ep.rendered_prompt));
            prompt.push_str(&format!("Response: {}\n", ep.response));
            if ep.was_suppressed {
                prompt.push_str("(This invocation was marked as unnecessary)\n");
            }
            prompt.push('\n');
        }
    }

    // === PREVIOUS CONVERSATION ===
    if compressed_history.is_some() || !recent_exchanges.is_empty() {
        prompt.push_str("=== PREVIOUS CONVERSATION ===\n");
        if let Some(history) = compressed_history {
            prompt.push_str(&format!("Summary: {}\n\n", history));
        }
        for exchange in recent_exchanges.iter().take(config.context_max_exchanges) {
            prompt.push_str(&format!("You previously said: {}\n", exchange.response));
        }
        prompt.push('\n');
    }

    // === AVAILABLE TOOLS ===
    prompt.push_str("=== AVAILABLE TOOLS ===\n");
    prompt.push_str("You have the following tools available:\n");
    for tool in &req.tool_definitions {
        prompt.push_str(&format!("- {}: {}\n", tool.name, tool.description));
    }
    prompt.push_str("\nUse suppress() if this invocation was not useful. Use recall() if you notice earlier events that should have triggered an invocation.\n");

    prompt
}

fn render_event_metadata(prompt: &mut String, event: &Event) {
    let m = &event.metadata;
    let keys: &[&str] = &[
        "author_id",
        "channel_id",
        "message_id",
        "guild_id",
        "user_id",
        "reply_to_message_id",
        "tool_name",
    ];
    let parts: Vec<String> = keys
        .iter()
        .filter_map(|&k| {
            m.get(k)
                .and_then(|v| v.as_str())
                .map(|v| format!("{k}={v}"))
        })
        .collect();
    if !parts.is_empty() {
        prompt.push_str(&format!("  ({})\n", parts.join(", ")));
    }
}
