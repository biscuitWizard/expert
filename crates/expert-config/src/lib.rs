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
    pub ollama_url: String,
    pub ollama_embeddings_url: String,
    pub llm_model: String,
    pub embeddings_model: String,
    pub orchestrator_url: String,

    pub embedding_dim: usize,
    pub ssm_hidden_dim: usize,
    pub ssm_max_k: usize,

    pub encoder_max_batch: usize,
    pub encoder_flush_ms: u64,

    pub fire_queue_ttl_ms: u64,
    pub debounce_ms: u64,
    pub refractory_ms: u64,
    pub refractory_dm_ms: u64,
    pub ssm_initial_theta: f32,

    pub context_max_recent_events: usize,
    pub context_max_episodes: usize,
    pub context_max_exchanges: usize,
    pub context_rag_timeout_ms: u64,

    pub llm_max_labels_per_invocation: usize,

    pub stream_maxlen: usize,

    // Session summarization
    pub exchange_summarize_threshold: usize,
    pub exchange_keep_after_summarize: usize,

    // Consensus scoring
    pub consensus_threshold: u32,
    pub consensus_time_window_ms: u64,

    // Training / adaptation
    pub checkpoint_dir: String,
    pub training_interval_secs: u64,
    pub training_label_threshold: u64,
    pub training_batch_size: usize,
    pub training_epochs: usize,
    pub training_learning_rate: f32,
    pub training_gradient_clip: f32,
    pub medium_label_threshold: u64,
    pub medium_batch_size: usize,
    pub medium_learning_rate: f32,

    // Cold-start scoring
    pub cold_start_dm_fires: u64,
    pub ssm_calibration_window: usize,
    pub ssm_calibration_threshold: f32,

    // Ollama warmup
    pub warmup_interval_secs: u64,
    pub warmup_keep_alive: String,

    // Debug panel
    pub panel_user: Option<String>,
    pub panel_pass: Option<String>,
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
            ollama_url: env_or("OLLAMA_URL", "http://127.0.0.1:11434"),
            ollama_embeddings_url: env_or("OLLAMA_EMBEDDINGS_URL", "http://127.0.0.1:11434"),
            llm_model: env_or("LLM_MODEL", "qwen3:32b"),
            embeddings_model: env_or("EMBEDDINGS_MODEL", "qwen3-embedding:8b"),
            orchestrator_url: env_or("ORCHESTRATOR_URL", "http://127.0.0.1:3000"),

            embedding_dim: env_parse("EMBEDDING_DIM", 4096),
            ssm_hidden_dim: env_parse("SSM_HIDDEN_DIM", 256),
            ssm_max_k: env_parse("SSM_MAX_K", 16),

            encoder_max_batch: env_parse("ENCODER_MAX_BATCH", 32),
            encoder_flush_ms: env_parse("ENCODER_FLUSH_MS", 10),

            fire_queue_ttl_ms: env_parse("FIRE_QUEUE_TTL_MS", 120_000),
            debounce_ms: env_parse("DEBOUNCE_MS", 500),
            refractory_ms: env_parse("REFRACTORY_MS", 5_000),
            refractory_dm_ms: env_parse("REFRACTORY_DM_MS", 1_000),
            ssm_initial_theta: env_parse("SSM_INITIAL_THETA", 0.15),

            context_max_recent_events: env_parse("CONTEXT_MAX_RECENT_EVENTS", 50),
            context_max_episodes: env_parse("CONTEXT_MAX_EPISODES", 5),
            context_max_exchanges: env_parse("CONTEXT_MAX_EXCHANGES", 3),
            context_rag_timeout_ms: env_parse("CONTEXT_RAG_TIMEOUT_MS", 2_000),

            llm_max_labels_per_invocation: env_parse("LLM_MAX_LABELS", 3),

            stream_maxlen: env_parse("STREAM_MAXLEN", 10_000),

            exchange_summarize_threshold: env_parse("EXCHANGE_SUMMARIZE_THRESHOLD", 10),
            exchange_keep_after_summarize: env_parse("EXCHANGE_KEEP_AFTER_SUMMARIZE", 3),

            consensus_threshold: env_parse("CONSENSUS_THRESHOLD", 2),
            consensus_time_window_ms: env_parse("CONSENSUS_TIME_WINDOW_MS", 30_000),

            checkpoint_dir: env_or("CHECKPOINT_DIR", "./checkpoints"),
            training_interval_secs: env_parse("TRAINING_INTERVAL_SECS", 3600),
            training_label_threshold: env_parse("TRAINING_LABEL_THRESHOLD", 100),
            training_batch_size: env_parse("TRAINING_BATCH_SIZE", 500),
            training_epochs: env_parse("TRAINING_EPOCHS", 3),
            training_learning_rate: env_parse("TRAINING_LEARNING_RATE", 0.001),
            training_gradient_clip: env_parse("TRAINING_GRADIENT_CLIP", 1.0),
            medium_label_threshold: env_parse("MEDIUM_LABEL_THRESHOLD", 20),
            medium_batch_size: env_parse("MEDIUM_BATCH_SIZE", 10),
            medium_learning_rate: env_parse("MEDIUM_LEARNING_RATE", 0.01),

            cold_start_dm_fires: env_parse("COLD_START_DM_FIRES", 20),
            ssm_calibration_window: env_parse("SSM_CALIBRATION_WINDOW", 50),
            ssm_calibration_threshold: env_parse("SSM_CALIBRATION_THRESHOLD", 0.7),

            warmup_interval_secs: env_parse("WARMUP_INTERVAL_SECS", 120),
            warmup_keep_alive: env_or("WARMUP_KEEP_ALIVE", "30m"),

            panel_user: env::var("PANEL_USER").ok(),
            panel_pass: env::var("PANEL_PASS").ok(),
        }
    }
}

pub fn init_tracing() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
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
        .map(|v| v.parse().unwrap_or_else(|_| panic!("{key} must be valid")))
        .unwrap_or(default)
}
