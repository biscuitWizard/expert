use std::env;
use tracing_subscriber::EnvFilter;

/// Central configuration for all Expert services. Loaded from environment
/// variables with sensible defaults for local development (docker-compose).
#[derive(Debug, Clone)]
pub struct Config {
    pub redis_url: String,
    pub qdrant_url: String,
    pub qdrant_grpc_port: u16,
    pub postgres_url: String,
    pub llamacpp_url: String,
    pub llamacpp_embeddings_url: String,
    pub orchestrator_url: String,

    pub embedding_dim: usize,
    pub ssm_hidden_dim: usize,

    pub encoder_max_batch: usize,
    pub encoder_flush_ms: u64,

    pub fire_queue_ttl_ms: u64,
    pub debounce_ms: u64,
    pub refractory_ms: u64,

    pub context_max_recent_events: usize,
    pub context_max_episodes: usize,
    pub context_max_exchanges: usize,
    pub context_rag_timeout_ms: u64,

    pub llm_max_labels_per_invocation: usize,

    pub stream_maxlen: usize,
}

impl Config {
    pub fn from_env() -> Self {
        Self {
            redis_url: env_or("REDIS_URL", "redis://127.0.0.1:6379"),
            qdrant_url: env_or("QDRANT_URL", "http://127.0.0.1:6334"),
            qdrant_grpc_port: env_or("QDRANT_GRPC_PORT", "6334")
                .parse()
                .expect("QDRANT_GRPC_PORT must be u16"),
            postgres_url: env_or(
                "DATABASE_URL",
                "postgres://expert:expert_dev@127.0.0.1:5432/expert_training",
            ),
            llamacpp_url: env_or("LLAMACPP_URL", "http://127.0.0.1:8080"),
            llamacpp_embeddings_url: env_or(
                "LLAMACPP_EMBEDDINGS_URL",
                "http://127.0.0.1:8081",
            ),
            orchestrator_url: env_or("ORCHESTRATOR_URL", "http://127.0.0.1:3000"),

            embedding_dim: env_parse("EMBEDDING_DIM", 4096),
            ssm_hidden_dim: env_parse("SSM_HIDDEN_DIM", 256),

            encoder_max_batch: env_parse("ENCODER_MAX_BATCH", 32),
            encoder_flush_ms: env_parse("ENCODER_FLUSH_MS", 10),

            fire_queue_ttl_ms: env_parse("FIRE_QUEUE_TTL_MS", 30_000),
            debounce_ms: env_parse("DEBOUNCE_MS", 500),
            refractory_ms: env_parse("REFRACTORY_MS", 5_000),

            context_max_recent_events: env_parse("CONTEXT_MAX_RECENT_EVENTS", 50),
            context_max_episodes: env_parse("CONTEXT_MAX_EPISODES", 5),
            context_max_exchanges: env_parse("CONTEXT_MAX_EXCHANGES", 3),
            context_rag_timeout_ms: env_parse("CONTEXT_RAG_TIMEOUT_MS", 2_000),

            llm_max_labels_per_invocation: env_parse("LLM_MAX_LABELS", 3),

            stream_maxlen: env_parse("STREAM_MAXLEN", 10_000),
        }
    }
}

pub fn init_tracing() {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(true)
        .compact()
        .init();
}

fn env_or(key: &str, default: &str) -> String {
    env::var(key).unwrap_or_else(|_| default.to_string())
}

fn env_parse<T: std::str::FromStr>(key: &str, default: T) -> T
where
    T::Err: std::fmt::Debug,
{
    env::var(key)
        .ok()
        .map(|v| v.parse().expect(&format!("{key} must be valid")))
        .unwrap_or(default)
}
