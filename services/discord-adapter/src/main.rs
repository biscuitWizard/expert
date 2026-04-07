mod actions;
mod auth;
mod events;
mod gateway;
mod rest;
mod tools;

use anyhow::Result;
use tracing::{error, info};

use expert_config::Config;
use expert_redis::{ServiceLogger, StateStore, StreamProducer};

#[tokio::main]
async fn main() -> Result<()> {
    expert_config::init_tracing();
    let config = Config::from_env();

    let stream_id = std::env::var("STREAM_ID").unwrap_or_else(|_| "discord".to_string());
    let username = std::env::var("DISCORD_USERNAME").expect("DISCORD_USERNAME is required");
    let guild_ids: Vec<String> = std::env::var("DISCORD_GUILD_IDS")
        .unwrap_or_default()
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    info!(stream_id, username, guilds = ?guild_ids, "starting discord-adapter");

    // Authenticate with Discord
    let token = auth::authenticate().await?;
    info!("discord authentication successful");

    // Connect to Redis
    let conn = expert_redis::connect(&config.redis_url).await?;
    let producer = StreamProducer::new(conn.clone(), config.stream_maxlen);
    let state = StateStore::new(conn.clone());

    // Build Discord REST client
    let rest_client = rest::DiscordRestClient::new(&token);

    // Resolve our own user ID from token
    let self_user = rest_client.get_current_user().await?;
    let self_user_id = self_user["id"]
        .as_str()
        .expect("user object must have id")
        .to_string();
    info!(self_user_id, "resolved self user id");

    // Bootstrap activity via orchestrator
    let display_name = self_user["global_name"]
        .as_str()
        .or_else(|| self_user["username"].as_str())
        .map(|s| s.to_string());
    let activity_id = bootstrap_activity(
        &config,
        &stream_id,
        &username,
        &self_user_id,
        display_name.as_deref(),
    )
    .await?;
    info!(activity_id, "activity bootstrapped");

    // Spawn action consumer
    let action_rest = rest::DiscordRestClient::new(&token);
    let action_stream_id = stream_id.clone();
    let action_conn = conn.clone();
    let mut action_producer = producer.clone();
    let mut action_state = StateStore::new(conn.clone());
    let action_svc_log = ServiceLogger::new(producer.clone(), "discord-adapter");
    tokio::spawn(async move {
        if let Err(e) = actions::run_action_consumer(
            action_conn,
            &mut action_producer,
            &mut action_state,
            &action_rest,
            &action_stream_id,
            action_svc_log,
        )
        .await
        {
            error!(error = %e, "action consumer exited");
        }
    });

    // Connect to Discord Gateway and run event loop
    gateway::run_gateway(
        &token,
        &self_user_id,
        &username,
        &stream_id,
        &guild_ids,
        producer,
        state,
    )
    .await?;

    Ok(())
}

async fn bootstrap_activity(
    config: &Config,
    stream_id: &str,
    bot_username: &str,
    bot_user_id: &str,
    bot_display_name: Option<&str>,
) -> Result<String> {
    let tool_defs = tools::discord_tool_definitions();
    let client = reqwest::Client::new();

    let body = serde_json::json!({
        "stream_id": stream_id,
        "domain": "discord",
        "goals": [
            {
                "name": "Respond to direct messages",
                "description": "Monitor and respond to DMs directed at the bot user"
            },
            {
                "name": "Participate in conversations",
                "description": "Engage in guild channel conversations when relevant"
            }
        ],
        "tool_definitions": tool_defs,
        "bot_identity": {
            "username": bot_username,
            "user_id": bot_user_id,
            "display_name": bot_display_name,
        },
    });

    let url = format!("{}/activities", config.orchestrator_url);

    // Retry loop: orchestrator may not be ready yet
    for attempt in 1..=30 {
        match client.post(&url).json(&body).send().await {
            Ok(resp) if resp.status().is_success() => {
                let data: serde_json::Value = resp.json().await?;
                let activity_id = data["activity_id"]
                    .as_str()
                    .unwrap_or("unknown")
                    .to_string();
                return Ok(activity_id);
            }
            Ok(resp) => {
                let status = resp.status();
                let text = resp.text().await.unwrap_or_default();
                tracing::warn!(attempt, %status, body = %text, "orchestrator not ready, retrying");
            }
            Err(e) => {
                tracing::warn!(attempt, error = %e, "orchestrator unreachable, retrying");
            }
        }
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
    }

    anyhow::bail!("failed to bootstrap activity after 30 attempts");
}
