use anyhow::{Context, Result, bail};
use tracing::info;

const DISCORD_API_BASE: &str = "https://discord.com/api/v10";

/// Authenticate with Discord. Tries DISCORD_TOKEN first,
/// falls back to email/password login.
pub async fn authenticate() -> Result<String> {
    if let Ok(token) = std::env::var("DISCORD_TOKEN") {
        let token = token.trim().to_string();
        if !token.is_empty() {
            info!("using DISCORD_TOKEN env var");
            return Ok(token);
        }
    }

    let email =
        std::env::var("DISCORD_EMAIL").context("neither DISCORD_TOKEN nor DISCORD_EMAIL is set")?;
    let password = std::env::var("DISCORD_PASSWORD")
        .context("DISCORD_EMAIL is set but DISCORD_PASSWORD is missing")?;

    info!("authenticating with email/password");
    login_with_credentials(&email, &password).await
}

async fn login_with_credentials(email: &str, password: &str) -> Result<String> {
    let client = reqwest::Client::new();

    let body = serde_json::json!({
        "login": email,
        "password": password,
        "undelete": false,
    });

    let resp = client
        .post(format!("{DISCORD_API_BASE}/auth/login"))
        .header("Content-Type", "application/json")
        .json(&body)
        .send()
        .await
        .context("failed to send login request")?;

    let status = resp.status();
    let data: serde_json::Value = resp
        .json()
        .await
        .context("failed to parse login response")?;

    if !status.is_success() {
        let message = data["message"].as_str().unwrap_or("unknown error");
        bail!("discord login failed ({}): {}", status, message);
    }

    if data.get("mfa").and_then(|v| v.as_bool()).unwrap_or(false) {
        bail!(
            "discord account requires MFA; use DISCORD_TOKEN instead \
             (extract your token from browser dev tools)"
        );
    }

    if data.get("captcha_key").is_some() {
        bail!(
            "discord login requires CAPTCHA; use DISCORD_TOKEN instead \
             (extract your token from browser dev tools)"
        );
    }

    data["token"]
        .as_str()
        .map(|s| s.to_string())
        .context("login response missing token field")
}
