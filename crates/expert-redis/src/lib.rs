pub mod streams;
pub mod state;
pub mod names;

pub use streams::{StreamProducer, StreamConsumer};
pub use state::StateStore;

use redis::Client;

pub async fn connect(redis_url: &str) -> Result<redis::aio::MultiplexedConnection, redis::RedisError> {
    let client = Client::open(redis_url)?;
    client.get_multiplexed_tokio_connection().await
}
