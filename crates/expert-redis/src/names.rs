//! Stream naming constants matching the schema in expert-redis README.

pub fn events_raw(stream_id: &str) -> String {
    format!("events.raw.{stream_id}")
}

pub fn events_embedded(stream_id: &str) -> String {
    format!("events.embedded.{stream_id}")
}

pub fn events_exchange(activity_id: &str) -> String {
    format!("events.exchange.{activity_id}")
}

pub fn actions(stream_id: &str) -> String {
    format!("actions.{stream_id}")
}

pub fn commands_worker(worker_id: &str) -> String {
    format!("commands.worker.{worker_id}")
}

pub const SIGNALS_FIRE: &str = "signals.fire";
pub const REQUESTS_CONTEXT: &str = "requests.context";
pub const REQUESTS_ENCODE: &str = "requests.encode";
pub const RESULTS_ENCODE: &str = "results.encode";
pub const QUERIES_RAG: &str = "queries.rag";
pub const RESULTS_RAG: &str = "results.rag";
pub const REQUESTS_SUMMARIZE: &str = "requests.summarize";
pub const RESULTS_SUMMARIZE: &str = "results.summarize";
pub const REQUESTS_GOAL_UPDATE: &str = "requests.goal_update";
pub const PACKAGES_READY: &str = "packages.ready";
pub const LABELS_WRITE: &str = "labels.write";
pub const EPISODES_WRITE: &str = "episodes.write";
pub const GOALS_WRITE: &str = "goals.write";
pub const CHECKPOINTS_AVAILABLE: &str = "checkpoints.available";
pub const EXCHANGES_ALL: &str = "exchanges.all";
pub const REQUESTS_TRAINING_BATCH: &str = "requests.training_batch";
pub const RESULTS_TRAINING_BATCH: &str = "results.training_batch";
pub const REQUESTS_FEWSHOT: &str = "requests.fewshot";
pub const RESULTS_FEWSHOT: &str = "results.fewshot";
pub const REQUESTS_FILTER_UPDATE: &str = "requests.filter_update";
pub const SIGNALS_INVOCATION_COMPLETE: &str = "signals.invocation_complete";
pub const SELF_KNOWLEDGE_WRITE: &str = "self_knowledge.write";
pub const LOGS_SERVICE: &str = "logs.service";

/// Redis key patterns for state store.
pub fn state_key(activity_id: &str) -> String {
    format!("state:{activity_id}")
}

pub fn seq_key(stream_id: &str) -> String {
    format!("seq:{stream_id}")
}

pub fn assignment_key(activity_id: &str) -> String {
    format!("assignment:{activity_id}")
}

pub fn fire_queue_key(activity_id: &str) -> String {
    format!("fire_queue:{activity_id}")
}

pub fn exchanges_key(activity_id: &str) -> String {
    format!("exchanges:{activity_id}")
}

pub fn history_key(activity_id: &str) -> String {
    format!("history:{activity_id}")
}

pub fn summarize_pending_key(activity_id: &str) -> String {
    format!("summarize_pending:{activity_id}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_events_raw_format() {
        assert_eq!(events_raw("mud-01"), "events.raw.mud-01");
    }

    #[test]
    fn test_events_embedded_format() {
        assert_eq!(events_embedded("mud-01"), "events.embedded.mud-01");
    }

    #[test]
    fn test_events_exchange_format() {
        assert_eq!(events_exchange("act-123"), "events.exchange.act-123");
    }

    #[test]
    fn test_actions_format() {
        assert_eq!(actions("mud-01"), "actions.mud-01");
    }

    #[test]
    fn test_commands_worker_format() {
        assert_eq!(commands_worker("worker-1"), "commands.worker.worker-1");
    }

    #[test]
    fn test_state_key_format() {
        assert_eq!(state_key("act-abc"), "state:act-abc");
    }

    #[test]
    fn test_seq_key_format() {
        assert_eq!(seq_key("mud-01"), "seq:mud-01");
    }

    #[test]
    fn test_assignment_key_format() {
        assert_eq!(assignment_key("act-abc"), "assignment:act-abc");
    }

    #[test]
    fn test_fire_queue_key_format() {
        assert_eq!(fire_queue_key("act-abc"), "fire_queue:act-abc");
    }

    #[test]
    fn test_constants_nonempty() {
        assert!(!SIGNALS_FIRE.is_empty());
        assert!(!REQUESTS_CONTEXT.is_empty());
        assert!(!REQUESTS_ENCODE.is_empty());
        assert!(!RESULTS_ENCODE.is_empty());
        assert!(!QUERIES_RAG.is_empty());
        assert!(!RESULTS_RAG.is_empty());
        assert!(!REQUESTS_SUMMARIZE.is_empty());
        assert!(!RESULTS_SUMMARIZE.is_empty());
        assert!(!REQUESTS_GOAL_UPDATE.is_empty());
        assert!(!PACKAGES_READY.is_empty());
        assert!(!LABELS_WRITE.is_empty());
        assert!(!EPISODES_WRITE.is_empty());
        assert!(!GOALS_WRITE.is_empty());
        assert!(!CHECKPOINTS_AVAILABLE.is_empty());
        assert!(!EXCHANGES_ALL.is_empty());
        assert!(!REQUESTS_TRAINING_BATCH.is_empty());
        assert!(!RESULTS_TRAINING_BATCH.is_empty());
        assert!(!REQUESTS_FEWSHOT.is_empty());
        assert!(!RESULTS_FEWSHOT.is_empty());
        assert!(!REQUESTS_FILTER_UPDATE.is_empty());
        assert!(!SIGNALS_INVOCATION_COMPLETE.is_empty());
        assert!(!SELF_KNOWLEDGE_WRITE.is_empty());
    }

    #[test]
    fn test_constants_dotted_namespace() {
        for name in [
            SIGNALS_FIRE,
            REQUESTS_CONTEXT,
            REQUESTS_ENCODE,
            RESULTS_ENCODE,
            QUERIES_RAG,
            RESULTS_RAG,
            REQUESTS_SUMMARIZE,
            RESULTS_SUMMARIZE,
            REQUESTS_GOAL_UPDATE,
            PACKAGES_READY,
            LABELS_WRITE,
            EPISODES_WRITE,
            GOALS_WRITE,
            CHECKPOINTS_AVAILABLE,
            EXCHANGES_ALL,
            REQUESTS_TRAINING_BATCH,
            RESULTS_TRAINING_BATCH,
            REQUESTS_FEWSHOT,
            RESULTS_FEWSHOT,
            REQUESTS_FILTER_UPDATE,
            SIGNALS_INVOCATION_COMPLETE,
            SELF_KNOWLEDGE_WRITE,
        ] {
            assert!(name.contains('.'), "{name} should use dotted namespace");
        }
    }

    #[test]
    fn test_exchanges_key_format() {
        assert_eq!(exchanges_key("act-abc"), "exchanges:act-abc");
    }

    #[test]
    fn test_history_key_format() {
        assert_eq!(history_key("act-abc"), "history:act-abc");
    }

    #[test]
    fn test_summarize_pending_key_format() {
        assert_eq!(
            summarize_pending_key("act-abc"),
            "summarize_pending:act-abc"
        );
    }
}
