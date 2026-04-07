mod llamacpp;
mod tools;

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::Result;
use tracing::{error, info, warn};

use expert_config::Config;
use expert_redis::names;
use expert_redis::{StreamConsumer, StreamProducer};
use expert_types::context::{ContextPackage, Episode, Exchange, ToolCall};

use llamacpp::LlamaCppClient;
use tools::ToolRouter;

#[tokio::main]
async fn main() -> Result<()> {
    expert_config::init_tracing();
    let config = Config::from_env();
    info!("starting llm-gateway");

    let conn = expert_redis::connect(&config.redis_url).await?;
    let mut producer = StreamProducer::new(conn.clone(), config.stream_maxlen);

    let mut consumer = StreamConsumer::new(
        conn.clone(),
        names::PACKAGES_READY.to_string(),
        "llm".to_string(),
        "llm-0".to_string(),
        500,
    )
    .await?;

    let client = LlamaCppClient::new(&config.llamacpp_url);

    loop {
        match consumer.consume::<ContextPackage>().await {
            Ok(Some((id, package))) => {
                let _ = consumer.ack(&id).await;
                info!(
                    activity_id = %package.activity_id,
                    goals = package.firing_goals.len(),
                    "received context package, invoking LLM"
                );

                match invoke_llm(&client, &package, &config, &mut producer).await {
                    Ok(_) => {
                        info!(activity_id = %package.activity_id, "LLM invocation complete");
                    }
                    Err(e) => {
                        error!(error = %e, activity_id = %package.activity_id, "LLM invocation failed");
                    }
                }
            }
            Ok(None) => {}
            Err(e) => {
                warn!(error = %e, "package consumer error");
                tokio::time::sleep(Duration::from_secs(1)).await;
            }
        }
    }
}

async fn invoke_llm(
    client: &LlamaCppClient,
    package: &ContextPackage,
    config: &Config,
    producer: &mut StreamProducer,
) -> Result<()> {
    let mut router = ToolRouter::new(package, config.llm_max_labels_per_invocation);

    // Build tool definitions for llamacpp
    let tools_json = tools::build_tools_json(&package.tool_definitions);

    // Initial LLM call
    let mut messages = vec![
        serde_json::json!({"role": "system", "content": "You are an AI agent monitoring a live event stream. Respond helpfully and use your tools to provide feedback on invocation quality."}),
        serde_json::json!({"role": "user", "content": package.rendered_prompt}),
    ];

    let mut all_tool_calls = Vec::new();
    let mut final_response = String::new();

    // Tool loop: keep calling until no more tool calls
    for iteration in 0..10 {
        let response = client.chat_completion(&messages, &tools_json).await?;

        if let Some(content) = response.get("content").and_then(|c| c.as_str()) {
            if !content.is_empty() {
                final_response = content.to_string();
            }
        }

        // Check for tool calls
        let tool_calls = response
            .get("tool_calls")
            .and_then(|tc| tc.as_array())
            .cloned()
            .unwrap_or_default();

        if tool_calls.is_empty() {
            // No more tool calls, done
            if final_response.is_empty() {
                if let Some(content) = response.get("content").and_then(|c| c.as_str()) {
                    final_response = content.to_string();
                }
            }
            break;
        }

        // Add assistant message with tool calls
        messages.push(serde_json::json!({
            "role": "assistant",
            "content": final_response,
            "tool_calls": tool_calls,
        }));

        // Process each tool call
        for tc in &tool_calls {
            let tool_name = tc
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(|n| n.as_str())
                .unwrap_or("");
            let arguments: serde_json::Value = tc
                .get("function")
                .and_then(|f| f.get("arguments"))
                .and_then(|a| {
                    if a.is_string() {
                        serde_json::from_str(a.as_str().unwrap()).ok()
                    } else {
                        Some(a.clone())
                    }
                })
                .unwrap_or(serde_json::Value::Object(Default::default()));
            let tool_call_id = tc.get("id").and_then(|id| id.as_str()).unwrap_or("unknown");

            info!(tool = tool_name, iteration, "processing tool call");

            let result = router.execute(tool_name, &arguments, producer).await;

            all_tool_calls.push(ToolCall {
                tool_name: tool_name.to_string(),
                arguments: arguments.clone(),
                result: Some(result.clone()),
            });

            messages.push(serde_json::json!({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": serde_json::to_string(&result).unwrap_or_default(),
            }));
        }
    }

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;

    // Post-invocation: publish exchange
    let exchange = Exchange {
        timestamp: now,
        rendered_prompt: package.rendered_prompt.clone(),
        response: final_response.clone(),
        tool_calls: all_tool_calls.clone(),
    };

    let exchange_stream = names::events_exchange(&package.activity_id);
    let _ = producer.publish(&exchange_stream, &exchange).await;

    // Publish episode
    let episode = Episode {
        id: uuid::Uuid::new_v4().to_string(),
        activity_id: package.activity_id.clone(),
        goal_id: package
            .firing_goals
            .first()
            .map(|g| g.id.clone())
            .unwrap_or_default(),
        domain: package.firing_goals.first().and_then(|g| g.domain.clone()),
        embedding: package.trigger_event.embedding.clone().unwrap_or_default(),
        trigger_event_id: package.trigger_event.id.clone(),
        trigger_scores: package.trigger_scores.clone(),
        rendered_prompt: package.rendered_prompt.clone(),
        response: final_response,
        tool_calls: all_tool_calls,
        was_suppressed: router.was_suppressed(),
        recalled_event_indices: router.recalled_indices().to_vec(),
        created_at: now,
    };
    let _ = producer.publish(names::EPISODES_WRITE, &episode).await;

    Ok(())
}
