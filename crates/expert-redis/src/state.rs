use redis::aio::MultiplexedConnection;
use redis::{AsyncCommands, RedisResult};
use serde::{de::DeserializeOwned, Serialize};

/// Key-value state store backed by Redis. Used for activity state
/// serialization, worker assignments, fire queue entries, and sequence counters.
pub struct StateStore {
    conn: MultiplexedConnection,
}

impl StateStore {
    pub fn new(conn: MultiplexedConnection) -> Self {
        Self { conn }
    }

    pub async fn get_json<T: DeserializeOwned>(&mut self, key: &str) -> RedisResult<Option<T>> {
        let raw: Option<String> = self.conn.get(key).await?;
        match raw {
            Some(json) => {
                let val = serde_json::from_str(&json).map_err(|e| {
                    redis::RedisError::from((
                        redis::ErrorKind::IoError,
                        "JSON deserialization failed",
                        e.to_string(),
                    ))
                })?;
                Ok(Some(val))
            }
            None => Ok(None),
        }
    }

    pub async fn set_json<T: Serialize>(&mut self, key: &str, val: &T) -> RedisResult<()> {
        let json = serde_json::to_string(val).map_err(|e| {
            redis::RedisError::from((
                redis::ErrorKind::IoError,
                "JSON serialization failed",
                e.to_string(),
            ))
        })?;
        self.conn.set(key, json).await
    }

    pub async fn set_json_ex<T: Serialize>(
        &mut self,
        key: &str,
        val: &T,
        expire_secs: u64,
    ) -> RedisResult<()> {
        let json = serde_json::to_string(val).map_err(|e| {
            redis::RedisError::from((
                redis::ErrorKind::IoError,
                "JSON serialization failed",
                e.to_string(),
            ))
        })?;
        self.conn.set_ex(key, json, expire_secs).await
    }

    pub async fn del(&mut self, key: &str) -> RedisResult<()> {
        self.conn.del(key).await
    }

    pub async fn set_str(&mut self, key: &str, val: &str) -> RedisResult<()> {
        self.conn.set(key, val).await
    }

    pub async fn get_str(&mut self, key: &str) -> RedisResult<Option<String>> {
        self.conn.get(key).await
    }

    /// Atomic increment, used for per-stream sequence counters.
    pub async fn incr(&mut self, key: &str) -> RedisResult<u64> {
        self.conn.incr(key, 1u64).await
    }
}

impl Clone for StateStore {
    fn clone(&self) -> Self {
        Self {
            conn: self.conn.clone(),
        }
    }
}
