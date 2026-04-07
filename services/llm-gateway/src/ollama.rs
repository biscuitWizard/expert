use anyhow::Result;
use async_trait::async_trait;
use serde_json::Value;

#[async_trait]
pub trait LlmClient: Send + Sync {
    async fn chat_completion(&self, messages: &[Value], tools: &[Value]) -> Result<Value>;
    async fn summarize(&self, text: &str) -> Result<String>;
}

pub struct OllamaClient {
    client: reqwest::Client,
    url: String,
    model: String,
}

impl OllamaClient {
    pub fn new(base_url: &str, model: &str) -> Self {
        Self {
            client: reqwest::Client::new(),
            url: format!("{}/v1/chat/completions", base_url.trim_end_matches('/')),
            model: model.to_string(),
        }
    }
}

#[async_trait]
impl LlmClient for OllamaClient {
    async fn chat_completion(&self, messages: &[Value], tools: &[Value]) -> Result<Value> {
        let mut body = serde_json::json!({
            "model": self.model,
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

        let message = json
            .get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("message"))
            .cloned()
            .unwrap_or_else(|| {
                serde_json::json!({"content": json.get("content").cloned().unwrap_or(Value::Null)})
            });

        Ok(message)
    }

    async fn summarize(&self, text: &str) -> Result<String> {
        let messages = vec![
            serde_json::json!({
                "role": "system",
                "content": "You are a concise summarizer. Compress the following session transcript into a brief narrative preserving key decisions, feedback actions, and context. Output only the summary."
            }),
            serde_json::json!({
                "role": "user",
                "content": text
            }),
        ];

        let body = serde_json::json!({
            "model": self.model,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 1024,
        });

        let resp = self
            .client
            .post(&self.url)
            .json(&body)
            .send()
            .await?
            .error_for_status()?;

        let json: Value = resp.json().await?;

        let content = json
            .get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .unwrap_or("")
            .to_string();

        Ok(content)
    }
}
