mod qdrant_store;

use std::time::Duration;

use anyhow::Result;
use redis::AsyncCommands;
use serde::{Deserialize, Serialize};
use tracing::{error, info, warn};

use expert_config::Config;
use expert_redis::names;
use expert_redis::{StreamConsumer, StreamProducer};
use expert_types::context::{ActivityExchange, Episode, Exchange};
use expert_types::goal::Goal;
use expert_types::signals::{SummarizeRequest, SummarizeResult};

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

        let mut producer = StreamProducer::new(conn_clone.clone(), maxlen);
        let mut kv_conn = conn_clone;

        loop {
            match consumer.consume::<RagQuery>().await {
                Ok(Some((id, query))) => {
                    let _ = consumer.ack(&id).await;
                    let result = handle_query(&store_clone, &query, &mut kv_conn).await;
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

    // Spawn exchange consumer on exchanges.all
    {
        let conn_ex = conn.clone();
        let producer_ex = StreamProducer::new(conn.clone(), config.stream_maxlen);
        let threshold = config.exchange_summarize_threshold;
        tokio::spawn(async move {
            if let Err(e) = run_exchange_consumer(conn_ex, producer_ex, threshold).await {
                error!(error = %e, "exchange consumer exited");
            }
        });
    }

    // Spawn summarize result consumer
    {
        let conn_sum = conn.clone();
        let keep = config.exchange_keep_after_summarize;
        tokio::spawn(async move {
            if let Err(e) = run_summarize_result_consumer(conn_sum, keep).await {
                error!(error = %e, "summarize result consumer exited");
            }
        });
    }

    info!("rag-service running");

    // Keep main alive
    loop {
        tokio::time::sleep(Duration::from_secs(60)).await;
    }
}

async fn run_exchange_consumer(
    conn: redis::aio::MultiplexedConnection,
    mut producer: StreamProducer,
    threshold: usize,
) -> Result<()> {
    let mut consumer = StreamConsumer::new(
        conn.clone(),
        names::EXCHANGES_ALL.to_string(),
        "rag-ex".to_string(),
        "rag-ex-0".to_string(),
        500,
    )
    .await?;

    let mut kv_conn = conn;

    loop {
        match consumer.consume::<ActivityExchange>().await {
            Ok(Some((id, ae))) => {
                let _ = consumer.ack(&id).await;

                let list_key = names::exchanges_key(&ae.activity_id);
                let json = serde_json::to_string(&ae.exchange).unwrap_or_default();
                let _: () = kv_conn.rpush(&list_key, &json).await?;

                let len: usize = kv_conn.llen(&list_key).await?;
                if len >= threshold {
                    let pending_key = names::summarize_pending_key(&ae.activity_id);
                    let already_pending: bool = kv_conn.exists(&pending_key).await?;
                    if !already_pending {
                        let raw_jsons: Vec<String> = kv_conn.lrange(&list_key, 0, -1).await?;
                        let raw_text = raw_jsons.join("\n---\n");

                        let req = SummarizeRequest {
                            activity_id: ae.activity_id.clone(),
                            session_id: uuid::Uuid::new_v4().to_string(),
                            raw_text,
                        };
                        if let Err(e) = producer.publish(names::REQUESTS_SUMMARIZE, &req).await {
                            error!(error = %e, "failed to publish summarize request");
                        } else {
                            let _: () = kv_conn.set_ex(&pending_key, "1", 120).await?;
                            info!(
                                activity_id = %ae.activity_id,
                                exchange_count = len,
                                "triggered session summarization"
                            );
                        }
                    }
                }
            }
            Ok(None) => {}
            Err(e) => {
                warn!(error = %e, "exchange consumer error");
                tokio::time::sleep(Duration::from_secs(1)).await;
            }
        }
    }
}

async fn run_summarize_result_consumer(
    conn: redis::aio::MultiplexedConnection,
    keep_after_summarize: usize,
) -> Result<()> {
    let mut consumer = StreamConsumer::new(
        conn.clone(),
        names::RESULTS_SUMMARIZE.to_string(),
        "rag-sum".to_string(),
        "rag-sum-0".to_string(),
        500,
    )
    .await?;

    let mut kv_conn = conn;

    loop {
        match consumer.consume::<SummarizeResult>().await {
            Ok(Some((id, result))) => {
                let _ = consumer.ack(&id).await;

                let history_key = names::history_key(&result.activity_id);
                let _: () = kv_conn
                    .set(&history_key, &result.compressed_narrative)
                    .await?;

                let list_key = names::exchanges_key(&result.activity_id);
                let total: isize = kv_conn.llen(&list_key).await?;
                let trim_from = total - keep_after_summarize as isize;
                if trim_from > 0 {
                    let _: () = kv_conn.ltrim(&list_key, trim_from, -1).await?;
                }

                let pending_key = names::summarize_pending_key(&result.activity_id);
                let _: () = kv_conn.del(&pending_key).await?;

                info!(
                    activity_id = %result.activity_id,
                    "session history compressed and stored"
                );
            }
            Ok(None) => {}
            Err(e) => {
                warn!(error = %e, "summarize result consumer error");
                tokio::time::sleep(Duration::from_secs(1)).await;
            }
        }
    }
}

async fn handle_query(
    store: &QdrantStore,
    query: &RagQuery,
    kv_conn: &mut redis::aio::MultiplexedConnection,
) -> RagResult {
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
            let activity_id = match &query.activity_id {
                Some(id) => id.as_str(),
                None => {
                    return RagResult {
                        request_id: query.request_id.clone(),
                        episodes: Vec::new(),
                        exchanges: Vec::new(),
                        compressed_history: None,
                        error: Some("missing activity_id for get_history".to_string()),
                    };
                }
            };

            let compressed: Option<String> = kv_conn
                .get(names::history_key(activity_id))
                .await
                .unwrap_or(None);

            let raw_jsons: Vec<String> = kv_conn
                .lrange(names::exchanges_key(activity_id), 0, -1)
                .await
                .unwrap_or_default();

            let exchanges: Vec<Exchange> = raw_jsons
                .iter()
                .filter_map(|j| serde_json::from_str(j).ok())
                .collect();

            RagResult {
                request_id: query.request_id.clone(),
                episodes: Vec::new(),
                exchanges,
                compressed_history: compressed,
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
