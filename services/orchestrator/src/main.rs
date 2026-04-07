mod registry;
mod api;
mod workers;

use std::sync::Arc;
use tokio::sync::RwLock;
use anyhow::Result;
use tracing::info;

use expert_config::Config;
use expert_redis::StreamProducer;
use registry::ActivityRegistry;

pub struct AppState {
    pub config: Config,
    pub registry: RwLock<ActivityRegistry>,
    pub producer: RwLock<StreamProducer>,
    pub state_store: RwLock<expert_redis::StateStore>,
}

#[tokio::main]
async fn main() -> Result<()> {
    expert_config::init_tracing();
    let config = Config::from_env();
    info!("starting orchestrator");

    let conn = expert_redis::connect(&config.redis_url).await?;
    let producer = StreamProducer::new(conn.clone(), config.stream_maxlen);
    let state_store = expert_redis::StateStore::new(conn.clone());

    let state = Arc::new(AppState {
        config,
        registry: RwLock::new(ActivityRegistry::new()),
        producer: RwLock::new(producer),
        state_store: RwLock::new(state_store),
    });

    // Spawn background workers for fire signal consumption and goal updates
    workers::spawn_fire_consumer(state.clone()).await;
    workers::spawn_goal_update_consumer(state.clone()).await;

    // Start HTTP server
    let app = api::router(state);
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
    info!("orchestrator REST API listening on :3000");
    axum::serve(listener, app).await?;

    Ok(())
}
