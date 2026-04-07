use std::collections::VecDeque;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;
use tokio::sync::{RwLock, broadcast};

const MAX_ENTRIES: usize = 1000;
const BROADCAST_CAPACITY: usize = 256;

#[derive(Debug, Clone, Serialize)]
pub struct PanelEvent {
    pub timestamp_ms: u64,
    pub kind: String,
    pub summary: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<serde_json::Value>,
}

pub struct EventLog {
    entries: RwLock<VecDeque<PanelEvent>>,
    tx: broadcast::Sender<PanelEvent>,
}

impl EventLog {
    pub fn new() -> Self {
        let (tx, _) = broadcast::channel(BROADCAST_CAPACITY);
        Self {
            entries: RwLock::new(VecDeque::with_capacity(MAX_ENTRIES)),
            tx,
        }
    }

    pub async fn push(
        &self,
        kind: impl Into<String>,
        summary: impl Into<String>,
        detail: Option<serde_json::Value>,
    ) {
        let event = PanelEvent {
            timestamp_ms: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            kind: kind.into(),
            summary: summary.into(),
            detail,
        };

        {
            let mut entries = self.entries.write().await;
            if entries.len() >= MAX_ENTRIES {
                entries.pop_front();
            }
            entries.push_back(event.clone());
        }

        let _ = self.tx.send(event);
    }

    pub async fn recent(&self, n: usize) -> Vec<PanelEvent> {
        let entries = self.entries.read().await;
        entries
            .iter()
            .rev()
            .take(n)
            .cloned()
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect()
    }

    pub fn subscribe(&self) -> broadcast::Receiver<PanelEvent> {
        self.tx.subscribe()
    }
}
