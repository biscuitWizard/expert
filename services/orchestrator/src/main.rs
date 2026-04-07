mod api;
pub mod event_log;
mod panel;
mod registry;
mod workers;

use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn};

use event_log::EventLog;
use expert_config::Config;
use expert_redis::StreamProducer;
use registry::ActivityRegistry;

#[derive(Debug, Clone, PartialEq)]
pub enum ModelWarmupStatus {
    Cold,
    Warming,
    Warm,
    Error(String),
}

pub struct AppState {
    pub config: Config,
    pub registry: RwLock<ActivityRegistry>,
    pub producer: RwLock<StreamProducer>,
    pub state_store: RwLock<expert_redis::StateStore>,
    pub event_log: Arc<EventLog>,
    pub warmup_status: RwLock<HashMap<String, ModelWarmupStatus>>,
}

#[tokio::main]
async fn main() -> Result<()> {
    expert_config::init_tracing();
    let config = Config::from_env();
    info!("starting orchestrator");

    let conn = expert_redis::connect(&config.redis_url).await?;
    let producer = StreamProducer::new(conn.clone(), config.stream_maxlen);
    let state_store = expert_redis::StateStore::new(conn.clone());
    let event_log = Arc::new(EventLog::new());

    let panel_enabled = config.panel_user.is_some() && config.panel_pass.is_some();

    let qdrant_client = if panel_enabled {
        match qdrant_client::Qdrant::from_url(&config.qdrant_url).build() {
            Ok(client) => {
                info!("qdrant client connected for debug panel");
                Some(client)
            }
            Err(e) => {
                warn!(error = %e, "failed to create qdrant client for panel — graph explorer will be unavailable");
                None
            }
        }
    } else {
        None
    };

    let mut initial_warmup = HashMap::new();
    initial_warmup.insert(config.llm_model.clone(), ModelWarmupStatus::Cold);
    initial_warmup.insert(config.embeddings_model.clone(), ModelWarmupStatus::Cold);

    let state = Arc::new(AppState {
        config,
        registry: RwLock::new(ActivityRegistry::new()),
        producer: RwLock::new(producer),
        state_store: RwLock::new(state_store),
        event_log: event_log.clone(),
        warmup_status: RwLock::new(initial_warmup),
    });

    // Spawn background workers
    workers::spawn_fire_consumer(state.clone()).await;
    workers::spawn_goal_update_consumer(state.clone()).await;
    workers::spawn_checkpoint_consumer(state.clone()).await;
    workers::spawn_threshold_feedback_task(state.clone()).await;
    workers::spawn_filter_update_consumer(state.clone()).await;
    workers::spawn_invocation_complete_consumer(state.clone()).await;
    workers::spawn_service_log_consumer(state.clone()).await;
    workers::spawn_warmup_task(state.clone()).await;

    // Build HTTP router
    let app = if panel_enabled {
        info!("debug panel enabled at /panel");
        event_log
            .push("panel_started", "Debug panel enabled", None)
            .await;
        api::router(state.clone()).merge(panel::router(state, qdrant_client))
    } else {
        api::router(state)
    };

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
    info!("orchestrator REST API listening on :3000");
    axum::serve(listener, app).await?;

    Ok(())
}
