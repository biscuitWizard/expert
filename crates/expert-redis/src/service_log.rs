use std::time::{SystemTime, UNIX_EPOCH};

use expert_types::service_log::{LogLevel, ServiceLogEntry};
use tracing::warn;

use crate::StreamProducer;
use crate::names;

/// Lightweight handle for publishing structured log entries to the
/// `logs.service` Redis stream. Clone-cheap (wraps a StreamProducer).
#[derive(Clone)]
pub struct ServiceLogger {
    producer: StreamProducer,
    service: String,
}

impl ServiceLogger {
    pub fn new(producer: StreamProducer, service: impl Into<String>) -> Self {
        Self {
            producer,
            service: service.into(),
        }
    }

    pub async fn error(&mut self, message: impl Into<String>, detail: Option<serde_json::Value>) {
        self.log(LogLevel::Error, message, detail).await;
    }

    pub async fn warn(&mut self, message: impl Into<String>, detail: Option<serde_json::Value>) {
        self.log(LogLevel::Warn, message, detail).await;
    }

    pub async fn info(&mut self, message: impl Into<String>, detail: Option<serde_json::Value>) {
        self.log(LogLevel::Info, message, detail).await;
    }

    async fn log(
        &mut self,
        level: LogLevel,
        message: impl Into<String>,
        detail: Option<serde_json::Value>,
    ) {
        let entry = ServiceLogEntry {
            service: self.service.clone(),
            level,
            message: message.into(),
            detail,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
        };
        if let Err(e) = self.producer.publish(names::LOGS_SERVICE, &entry).await {
            warn!(error = %e, "failed to publish service log entry");
        }
    }
}
