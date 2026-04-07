use anyhow::Result;
use serde_json::Value;

pub struct LlamaCppClient {
    client: reqwest::Client,
    url: String,
}

impl LlamaCppClient {
    pub fn new(base_url: &str) -> Self {
        Self {
            client: reqwest::Client::new(),
            url: format!("{}/v1/chat/completions", base_url.trim_end_matches('/')),
        }
    }

    pub async fn chat_completion(&self, messages: &[Value], tools: &[Value]) -> Result<Value> {
        let mut body = serde_json::json!({
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 2048,
        });

        if !tools.is_empty() {
            body["tools"] = Value::Array(tools.to_vec());
        }

        let resp = self
            .client
            .post(&self.url)
            .json(&body)
            .send()
            .await?
            .error_for_status()?;

        let json: Value = resp.json().await?;

        // Extract the first choice's message
        let message = json
            .get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("message"))
            .cloned()
            .unwrap_or_else(|| {
                // Fallback: some llamacpp versions return content directly
                serde_json::json!({"content": json.get("content").cloned().unwrap_or(Value::Null)})
            });

        Ok(message)
    }
}
