use redis::aio::MultiplexedConnection;
use redis::{AsyncCommands, RedisResult};
use serde::{de::DeserializeOwned, Serialize};
use tracing::{debug, warn};

const DATA_FIELD: &str = "data";

pub struct StreamProducer {
    conn: MultiplexedConnection,
    maxlen: usize,
}

impl StreamProducer {
    pub fn new(conn: MultiplexedConnection, maxlen: usize) -> Self {
        Self { conn, maxlen }
    }

    /// XADD with approximate MAXLEN trimming. Returns the stream entry ID.
    pub async fn publish<T: Serialize>(
        &mut self,
        stream: &str,
        msg: &T,
    ) -> RedisResult<String> {
        let json = serde_json::to_string(msg)
            .map_err(|e| redis::RedisError::from((
                redis::ErrorKind::IoError,
                "JSON serialization failed",
                e.to_string(),
            )))?;

        let id: String = redis::cmd("XADD")
            .arg(stream)
            .arg("MAXLEN")
            .arg("~")
            .arg(self.maxlen)
            .arg("*")
            .arg(DATA_FIELD)
            .arg(&json)
            .query_async(&mut self.conn)
            .await?;

        debug!(stream, id = %id, "published message");
        Ok(id)
    }
}

impl Clone for StreamProducer {
    fn clone(&self) -> Self {
        Self {
            conn: self.conn.clone(),
            maxlen: self.maxlen,
        }
    }
}

pub struct StreamConsumer {
    conn: MultiplexedConnection,
    stream: String,
    group: String,
    consumer: String,
    block_ms: usize,
}

impl StreamConsumer {
    /// Create a consumer and ensure the consumer group exists.
    pub async fn new(
        mut conn: MultiplexedConnection,
        stream: String,
        group: String,
        consumer: String,
        block_ms: usize,
    ) -> RedisResult<Self> {
        // Create group; ignore error if it already exists.
        let result: RedisResult<()> = redis::cmd("XGROUP")
            .arg("CREATE")
            .arg(&stream)
            .arg(&group)
            .arg("0")
            .arg("MKSTREAM")
            .query_async(&mut conn)
            .await;

        if let Err(ref e) = result {
            let msg = format!("{e}");
            if !msg.contains("BUSYGROUP") {
                return Err(result.unwrap_err());
            }
        }

        Ok(Self { conn, stream, group, consumer, block_ms })
    }

    /// Block-read one message from the stream. Returns `None` on timeout.
    pub async fn consume<T: DeserializeOwned>(
        &mut self,
    ) -> RedisResult<Option<(String, T)>> {
        let opts = redis::streams::StreamReadOptions::default()
            .group(&self.group, &self.consumer)
            .count(1)
            .block(self.block_ms);

        let result: redis::streams::StreamReadReply =
            self.conn.xread_options(&[&self.stream], &[">"], &opts).await?;

        for stream_key in result.keys {
            for entry in stream_key.ids {
                let id = entry.id.clone();
                if let Some(redis::Value::BulkString(bytes)) = entry.map.get(DATA_FIELD) {
                    let json_str = String::from_utf8_lossy(bytes);
                    match serde_json::from_str::<T>(&json_str) {
                        Ok(msg) => return Ok(Some((id, msg))),
                        Err(e) => {
                            warn!(stream = %self.stream, %id, error = %e, "deserialization failed, acking and skipping");
                            self.ack(&id).await?;
                        }
                    }
                }
            }
        }

        Ok(None)
    }

    pub async fn ack(&mut self, id: &str) -> RedisResult<()> {
        redis::cmd("XACK")
            .arg(&self.stream)
            .arg(&self.group)
            .arg(id)
            .query_async(&mut self.conn)
            .await
    }

    pub fn stream(&self) -> &str {
        &self.stream
    }
}

/// XREVRANGE helper for context-builder lookback.
/// Returns entries from `end` to `start` (reverse order), up to `count`.
pub async fn xrevrange<T: DeserializeOwned>(
    conn: &mut MultiplexedConnection,
    stream: &str,
    end: &str,
    start: &str,
    count: usize,
) -> RedisResult<Vec<(String, T)>> {
    let raw: Vec<redis::Value> = redis::cmd("XREVRANGE")
        .arg(stream)
        .arg(end)
        .arg(start)
        .arg("COUNT")
        .arg(count)
        .query_async(conn)
        .await?;

    let mut results = Vec::new();
    for entry_val in raw {
        if let redis::Value::Array(parts) = entry_val {
            if parts.len() >= 2 {
                let id = match &parts[0] {
                    redis::Value::BulkString(b) => String::from_utf8_lossy(b).to_string(),
                    _ => continue,
                };
                if let redis::Value::Array(ref fields) = parts[1] {
                    // Fields come as [key, value, key, value, ...]
                    let mut i = 0;
                    while i + 1 < fields.len() {
                        let key = match &fields[i] {
                            redis::Value::BulkString(b) => String::from_utf8_lossy(b).to_string(),
                            _ => { i += 2; continue; }
                        };
                        if key == DATA_FIELD {
                            if let redis::Value::BulkString(ref bytes) = fields[i + 1] {
                                let json_str = String::from_utf8_lossy(bytes);
                                if let Ok(msg) = serde_json::from_str::<T>(&json_str) {
                                    results.push((id.clone(), msg));
                                }
                            }
                        }
                        i += 2;
                    }
                }
            }
        }
    }

    Ok(results)
}
