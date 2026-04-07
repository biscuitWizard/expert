use anyhow::{Context, Result, bail};
use serde_json::Value;

const DISCORD_API: &str = "https://discord.com/api/v10";

pub struct DiscordRestClient {
    client: reqwest::Client,
    token: String,
}

impl DiscordRestClient {
    pub fn new(token: &str) -> Self {
        Self {
            client: reqwest::Client::new(),
            token: token.to_string(),
        }
    }

    fn auth_header(&self) -> String {
        self.token.clone()
    }

    async fn request(
        &self,
        method: reqwest::Method,
        path: &str,
        body: Option<&Value>,
    ) -> Result<Value> {
        let url = format!("{DISCORD_API}{path}");
        let mut req = self
            .client
            .request(method, &url)
            .header("Authorization", self.auth_header())
            .header("Content-Type", "application/json");

        if let Some(b) = body {
            req = req.json(b);
        }

        let resp = req.send().await.context("discord REST request failed")?;
        let status = resp.status();

        if status == reqwest::StatusCode::NO_CONTENT {
            return Ok(Value::Null);
        }

        let data: Value = resp
            .json()
            .await
            .context("failed to parse discord REST response")?;

        if !status.is_success() {
            let msg = data["message"].as_str().unwrap_or("unknown error");
            bail!("discord API error ({status}): {msg}");
        }

        Ok(data)
    }

    // --- User info ---

    pub async fn get_current_user(&self) -> Result<Value> {
        self.request(reqwest::Method::GET, "/users/@me", None).await
    }

    // --- Messaging ---

    pub async fn send_message(&self, channel_id: &str, content: &str) -> Result<Value> {
        let body = serde_json::json!({ "content": content });
        self.request(
            reqwest::Method::POST,
            &format!("/channels/{channel_id}/messages"),
            Some(&body),
        )
        .await
    }

    pub async fn reply_to_message(
        &self,
        channel_id: &str,
        message_id: &str,
        content: &str,
    ) -> Result<Value> {
        let body = serde_json::json!({
            "content": content,
            "message_reference": {
                "message_id": message_id,
            }
        });
        self.request(
            reqwest::Method::POST,
            &format!("/channels/{channel_id}/messages"),
            Some(&body),
        )
        .await
    }

    /// Open a DM channel with a user and send a message.
    pub async fn send_dm(&self, user_id: &str, content: &str) -> Result<Value> {
        let dm_channel = self.open_dm(user_id).await?;
        let channel_id = dm_channel["id"]
            .as_str()
            .context("DM channel response missing id")?;
        self.send_message(channel_id, content).await
    }

    async fn open_dm(&self, user_id: &str) -> Result<Value> {
        let body = serde_json::json!({ "recipient_id": user_id });
        self.request(reqwest::Method::POST, "/users/@me/channels", Some(&body))
            .await
    }

    // --- Reactions ---

    pub async fn react_to_message(
        &self,
        channel_id: &str,
        message_id: &str,
        emoji: &str,
    ) -> Result<()> {
        let encoded_emoji = urlencoding(emoji);
        self.request(
            reqwest::Method::PUT,
            &format!("/channels/{channel_id}/messages/{message_id}/reactions/{encoded_emoji}/@me"),
            None,
        )
        .await?;
        Ok(())
    }

    // --- Guilds ---

    pub async fn join_guild(&self, invite_code: &str) -> Result<Value> {
        let code = strip_invite_url(invite_code);
        self.request(reqwest::Method::POST, &format!("/invites/{code}"), None)
            .await
    }

    pub async fn leave_guild(&self, guild_id: &str) -> Result<()> {
        self.request(
            reqwest::Method::DELETE,
            &format!("/users/@me/guilds/{guild_id}"),
            None,
        )
        .await?;
        Ok(())
    }

    // --- Friends / Relationships ---

    pub async fn send_friend_request(&self, username: &str) -> Result<Value> {
        let body = serde_json::json!({ "username": username });
        self.request(
            reqwest::Method::POST,
            "/users/@me/relationships",
            Some(&body),
        )
        .await
    }

    pub async fn accept_friend_request(&self, user_id: &str) -> Result<()> {
        let body = serde_json::json!({});
        self.request(
            reqwest::Method::PUT,
            &format!("/users/@me/relationships/{user_id}"),
            Some(&body),
        )
        .await?;
        Ok(())
    }

    pub async fn remove_friend(&self, user_id: &str) -> Result<()> {
        self.request(
            reqwest::Method::DELETE,
            &format!("/users/@me/relationships/{user_id}"),
            None,
        )
        .await?;
        Ok(())
    }

    // --- Typing ---

    pub async fn trigger_typing(&self, channel_id: &str) -> Result<()> {
        self.request(
            reqwest::Method::POST,
            &format!("/channels/{channel_id}/typing"),
            None,
        )
        .await?;
        Ok(())
    }
}

/// Minimal URL-encoding for emoji strings (handles unicode emoji and custom :name:id format).
fn urlencoding(s: &str) -> String {
    url::form_urlencoded::byte_serialize(s.as_bytes()).collect()
}

/// Extract bare invite code from a full URL or bare code.
fn strip_invite_url(input: &str) -> &str {
    input
        .trim_start_matches("https://discord.gg/")
        .trim_start_matches("https://discord.com/invite/")
        .trim_start_matches("discord.gg/")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn urlencoding_unicode_emoji() {
        let encoded = urlencoding("👍");
        assert!(!encoded.contains('👍'));
        assert!(!encoded.is_empty());
    }

    #[test]
    fn urlencoding_ascii_emoji_name() {
        let encoded = urlencoding("thumbsup");
        assert_eq!(encoded, "thumbsup");
    }

    #[test]
    fn urlencoding_custom_emoji() {
        let encoded = urlencoding("custom:123456");
        assert!(encoded.contains("custom"));
    }

    #[test]
    fn strip_invite_bare_code() {
        assert_eq!(strip_invite_url("abc123"), "abc123");
    }

    #[test]
    fn strip_invite_discord_gg() {
        assert_eq!(strip_invite_url("https://discord.gg/abc123"), "abc123");
    }

    #[test]
    fn strip_invite_discord_com() {
        assert_eq!(
            strip_invite_url("https://discord.com/invite/abc123"),
            "abc123"
        );
    }

    #[test]
    fn strip_invite_no_https() {
        assert_eq!(strip_invite_url("discord.gg/abc123"), "abc123");
    }
}
