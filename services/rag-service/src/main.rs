mod qdrant_store;

use std::time::Duration;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tracing::{error, info, warn};

use expert_config::Config;
use expert_redis::names;
use expert_redis::{StreamConsumer, StreamProducer};
use expert_types::context::{Episode, Exchange};
use expert_types::goal::Goal;

use qdrant_store::QdrantStore;

#[derive(Deserialize)]
struct RagQuery {
    request_id: String,
    query_type: String,
    #[serde(default)]
    embedding: Option<Vec<f32>>,
    #[serde(default)]
    activity_id: Option<String>,
    #[serde(default)]
    k: Option<usize>,
}

#[derive(Serialize)]
struct RagResult {
    request_id: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    episodes: Vec<Episode>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    exchanges: Vec<Exchange>,
    #[serde(skip_serializing_if = "Option::is_none")]
    compressed_history: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    expert_config::init_tracing();
    let config = Config::from_env();
    info!("starting rag-service");

    // Initialize Qdrant
    let store = QdrantStore::new(&config).await?;
    info!("qdrant collections initialized");

    let conn = expert_redis::connect(&config.redis_url).await?;
    let _producer = StreamProducer::new(conn.clone(), config.stream_maxlen);

    // Spawn query consumer
    let store_clone = store.clone();
    let conn_clone = conn.clone();
    let maxlen = config.stream_maxlen;
    tokio::spawn(async move {
        let mut consumer = match StreamConsumer::new(
            conn_clone.clone(),
            names::QUERIES_RAG.to_string(),
            "rag".to_string(),
            "rag-0".to_string(),
            500,
        )
        .await
        {
            Ok(c) => c,
            Err(e) => {
                error!(error = %e, "failed to create RAG query consumer");
                return;
            }
        };

        let mut producer = StreamProducer::new(conn_clone, maxlen);

        loop {
            match consumer.consume::<RagQuery>().await {
                Ok(Some((id, query))) => {
                    let _ = consumer.ack(&id).await;
                    let result = handle_query(&store_clone, &query).await;
                    if let Err(e) = producer.publish(names::RESULTS_RAG, &result).await {
                        error!(error = %e, "failed to publish RAG result");
                    }
                }
                Ok(None) => {}
                Err(e) => {
                    warn!(error = %e, "RAG query consumer error");
                    tokio::time::sleep(Duration::from_secs(1)).await;
                }
            }
        }
    });

    // Spawn goals.write consumer
    let store_goals = store.clone();
    let conn_goals = conn.clone();
    tokio::spawn(async move {
        let mut consumer = match StreamConsumer::new(
            conn_goals,
            names::GOALS_WRITE.to_string(),
            "rag-goals".to_string(),
            "rag-goals-0".to_string(),
            500,
        )
        .await
        {
            Ok(c) => c,
            Err(e) => {
                error!(error = %e, "failed to create goals consumer");
                return;
            }
        };

        loop {
            match consumer.consume::<Goal>().await {
                Ok(Some((id, goal))) => {
                    let _ = consumer.ack(&id).await;
                    if let Err(e) = store_goals.upsert_goal(&goal).await {
                        warn!(error = %e, goal_id = %goal.id, "failed to upsert goal");
                    }
                }
                Ok(None) => {}
                Err(e) => {
                    warn!(error = %e, "goals consumer error");
                    tokio::time::sleep(Duration::from_secs(1)).await;
                }
            }
        }
    });

    // Spawn episodes.write consumer
    let store_episodes = store.clone();
    let conn_episodes = conn.clone();
    tokio::spawn(async move {
        let mut consumer = match StreamConsumer::new(
            conn_episodes,
            names::EPISODES_WRITE.to_string(),
            "rag-episodes".to_string(),
            "rag-episodes-0".to_string(),
            500,
        )
        .await
        {
            Ok(c) => c,
            Err(e) => {
                error!(error = %e, "failed to create episodes consumer");
                return;
            }
        };

        loop {
            match consumer.consume::<Episode>().await {
                Ok(Some((id, episode))) => {
                    let _ = consumer.ack(&id).await;
                    if let Err(e) = store_episodes.insert_episode(&episode).await {
                        warn!(error = %e, episode_id = %episode.id, "failed to insert episode");
                    } else {
                        info!(episode_id = %episode.id, "episode stored");
                    }
                }
                Ok(None) => {}
                Err(e) => {
                    warn!(error = %e, "episodes consumer error");
                    tokio::time::sleep(Duration::from_secs(1)).await;
                }
            }
        }
    });

    // Spawn events.exchange.* consumer (for MVP: log and store in Redis list)
    // In production this would handle multiple activity_id streams dynamically
    info!("rag-service running (exchange consumer deferred to per-activity dynamic subscription)");

    // Keep main alive
    loop {
        tokio::time::sleep(Duration::from_secs(60)).await;
    }
}

async fn handle_query(store: &QdrantStore, query: &RagQuery) -> RagResult {
    match query.query_type.as_str() {
        "semantic_search" => {
            let embedding = match &query.embedding {
                Some(e) => e.clone(),
                None => {
                    return RagResult {
                        request_id: query.request_id.clone(),
                        episodes: Vec::new(),
                        exchanges: Vec::new(),
                        compressed_history: None,
                        error: Some("missing embedding for semantic search".to_string()),
                    };
                }
            };

            let k = query.k.unwrap_or(5);
            match store.search_episodes(&embedding, k).await {
                Ok(episodes) => RagResult {
                    request_id: query.request_id.clone(),
                    episodes,
                    exchanges: Vec::new(),
                    compressed_history: None,
                    error: None,
                },
                Err(e) => RagResult {
                    request_id: query.request_id.clone(),
                    episodes: Vec::new(),
                    exchanges: Vec::new(),
                    compressed_history: None,
                    error: Some(format!("search failed: {e}")),
                },
            }
        }
        "get_history" => {
            // MVP: return empty history (session summarization deferred)
            RagResult {
                request_id: query.request_id.clone(),
                episodes: Vec::new(),
                exchanges: Vec::new(),
                compressed_history: None,
                error: None,
            }
        }
        "get_goal" => {
            // MVP: goals are managed by orchestrator, not queried from RAG
            RagResult {
                request_id: query.request_id.clone(),
                episodes: Vec::new(),
                exchanges: Vec::new(),
                compressed_history: None,
                error: None,
            }
        }
        other => RagResult {
            request_id: query.request_id.clone(),
            episodes: Vec::new(),
            exchanges: Vec::new(),
            compressed_history: None,
            error: Some(format!("unknown query type: {other}")),
        },
    }
}
