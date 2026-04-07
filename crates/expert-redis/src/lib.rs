pub mod names;
pub mod state;
pub mod streams;

pub use state::StateStore;
pub use streams::{StreamConsumer, StreamProducer};

use redis::Client;

pub async fn connect(
    redis_url: &str,
) -> Result<redis::aio::MultiplexedConnection, redis::RedisError> {
    let client = Client::open(redis_url)?;
    client.get_multiplexed_tokio_connection().await
}
