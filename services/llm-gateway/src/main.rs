mod ollama;
mod tools;

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::Result;
use tracing::{error, info, warn};

use expert_config::Config;
use expert_redis::names;
use expert_redis::{ServiceLogger, StateStore, StreamConsumer, StreamProducer};
use expert_types::context::{
    ActivityExchange, ContextPackage, ConversationTurn, Episode, Exchange, ToolCall,
};
use expert_types::signals::{InvocationComplete, SummarizeRequest, SummarizeResult};

use ollama::{LlmClient, OllamaClient};
use tools::ToolRouter;

#[tokio::main]
async fn main() -> Result<()> {
    expert_config::init_tracing();
    let config = Config::from_env();
    info!("starting llm-gateway");

    let conn = expert_redis::connect(&config.redis_url).await?;
    let producer = StreamProducer::new(conn.clone(), config.stream_maxlen);

    // Spawn summarization consumer
    {
        let conn = conn.clone();
        let mut sum_producer = producer.clone();
        let client = OllamaClient::new(&config.ollama_url, &config.llm_model);
        tokio::spawn(async move {
            if let Err(e) = run_summarize_consumer(conn, &client, &mut sum_producer).await {
                error!(error = %e, "summarize consumer exited");
            }
        });
    }

    let mut consumer = StreamConsumer::new(
        conn.clone(),
        names::PACKAGES_READY.to_string(),
        "llm".to_string(),
        "llm-0".to_string(),
        500,
    )
    .await?;

    let client = OllamaClient::new(&config.ollama_url, &config.llm_model);
    let mut producer = producer;
    let mut state = StateStore::new(conn.clone());
    let mut svc_log = ServiceLogger::new(producer.clone(), "llm-gateway");

    loop {
        match consumer.consume::<ContextPackage>().await {
            Ok(Some((id, package))) => {
                let _ = consumer.ack(&id).await;
                info!(
                    activity_id = %package.activity_id,
                    goals = package.firing_goals.len(),
                    "received context package, invoking LLM"
                );

                let result = tokio::time::timeout(
                    Duration::from_secs(120),
                    invoke_llm(&client, &package, &config, &mut producer, &mut state),
                )
                .await;
                let (success, result) = match result {
                    Ok(inner) => (inner.is_ok(), inner),
                    Err(_) => {
                        error!(
                            activity_id = %package.activity_id,
                            "LLM invocation timed out after 120s"
                        );
                        (false, Err(anyhow::anyhow!("invocation timed out")))
                    }
                };

                match &result {
                    Ok(_) => {
                        info!(activity_id = %package.activity_id, "LLM invocation complete");
                    }
                    Err(e) => {
                        error!(error = %e, activity_id = %package.activity_id, "LLM invocation failed");
                        svc_log
                            .error(
                                format!(
                                    "LLM invocation failed for {}: {e}",
                                    &package.activity_id[..8]
                                ),
                                Some(serde_json::json!({ "activity_id": package.activity_id })),
                            )
                            .await;
                    }
                }

                let response_preview = result.as_ref().ok().and_then(|resp| {
                    if resp.is_empty() {
                        None
                    } else {
                        let truncated: String = resp.chars().take(200).collect();
                        Some(truncated)
                    }
                });

                let event_type = package
                    .trigger_event
                    .metadata
                    .get("event_type")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                let author_name = package
                    .trigger_event
                    .metadata
                    .get("author_name")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());

                let complete = InvocationComplete {
                    activity_id: package.activity_id.clone(),
                    success,
                    response_preview,
                    event_type,
                    author_name,
                };
                if let Err(e) = producer
                    .publish(names::SIGNALS_INVOCATION_COMPLETE, &complete)
                    .await
                {
                    error!(error = %e, "failed to publish invocation complete signal");
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

async fn run_summarize_consumer(
    conn: redis::aio::MultiplexedConnection,
    client: &dyn LlmClient,
    producer: &mut StreamProducer,
) -> Result<()> {
    let mut consumer = StreamConsumer::new(
        conn,
        names::REQUESTS_SUMMARIZE.to_string(),
        "llm-sum".to_string(),
        "sum-0".to_string(),
        500,
    )
    .await?;

    loop {
        match consumer.consume::<SummarizeRequest>().await {
            Ok(Some((id, req))) => {
                let _ = consumer.ack(&id).await;
                info!(activity_id = %req.activity_id, "summarizing session history");

                match client.summarize(&req.raw_text).await {
                    Ok(narrative) => {
                        let result = SummarizeResult {
                            activity_id: req.activity_id,
                            session_id: req.session_id,
                            compressed_narrative: narrative,
                        };
                        if let Err(e) = producer.publish(names::RESULTS_SUMMARIZE, &result).await {
                            error!(error = %e, "failed to publish summarize result");
                        }
                    }
                    Err(e) => {
                        error!(error = %e, "summarization call failed");
                    }
                }
            }
            Ok(None) => {}
            Err(e) => {
                warn!(error = %e, "summarize consumer error");
                tokio::time::sleep(Duration::from_secs(1)).await;
            }
        }
    }
}

async fn invoke_llm(
    client: &dyn LlmClient,
    package: &ContextPackage,
    config: &Config,
    producer: &mut StreamProducer,
    state: &mut StateStore,
) -> Result<String> {
    let mut router = ToolRouter::new(package, config.llm_max_labels_per_invocation);

    // Build tool definitions for the LLM
    let tools_json = tools::build_tools_json(&package.tool_definitions);

    // Initial LLM call
    let mut messages = vec![
        serde_json::json!({"role": "system", "content": "You are Zero, an autonomous expert system. You observe live event streams, reason about what you see, and act when your goals demand it. Respond naturally and use your tools to provide feedback on invocation quality and to evolve your self-knowledge.\n\nBEHAVIORAL RULES:\n- Event metadata contains Discord snowflake IDs (author_id, channel_id, message_id, etc.). Use these values EXACTLY as provided -- never fabricate or guess IDs.\n- When the trigger event has event_type=dm, respond using the send_dm tool with the author_id from the event metadata.\n- When the trigger event has event_type=message (guild channel), respond using reply_to_message with the channel_id and message_id from the event metadata.\n- Events with is_self=true are your own previous messages. Do not respond to yourself.\n- Only call suppress() when the triggering event genuinely does not warrant engagement (e.g., background noise irrelevant to your goals). If someone is talking to you, respond."}),
        serde_json::json!({"role": "user", "content": package.rendered_prompt}),
    ];

    let mut all_tool_calls = Vec::new();
    let mut final_response = String::new();

    // Tool loop: keep calling until no more tool calls
    for iteration in 0..10 {
        let response = client.chat_completion(&messages, &tools_json).await?;

        if let Some(content) = response.get("content").and_then(|c| c.as_str())
            && !content.is_empty()
        {
            final_response = content.to_string();
        }

        // Check for tool calls
        let tool_calls = response
            .get("tool_calls")
            .and_then(|tc| tc.as_array())
            .cloned()
            .unwrap_or_default();

        if tool_calls.is_empty() {
            // No more tool calls, done
            if final_response.is_empty()
                && let Some(content) = response.get("content").and_then(|c| c.as_str())
            {
                final_response = content.to_string();
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

    let activity_exchange = ActivityExchange {
        activity_id: package.activity_id.clone(),
        exchange: exchange.clone(),
    };
    let _ = producer
        .publish(names::EXCHANGES_ALL, &activity_exchange)
        .await;

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
        response: final_response.clone(),
        tool_calls: all_tool_calls,
        was_suppressed: router.was_suppressed(),
        recalled_event_indices: router.recalled_indices().to_vec(),
        created_at: now,
    };
    let _ = producer.publish(names::EPISODES_WRITE, &episode).await;

    // Push conversation turns to per-channel buffer for multi-turn coherence
    if let Some(channel_id) = package
        .trigger_event
        .metadata
        .get("channel_id")
        .and_then(|v| v.as_str())
    {
        let conv_key = names::conversation_key(&package.trigger_event.stream_id, channel_id);
        let user_turn = ConversationTurn {
            role: "user".to_string(),
            content: package.trigger_event.raw.clone(),
            timestamp: package.trigger_event.timestamp,
        };
        let _ = state
            .list_push_capped(&conv_key, &user_turn, 20, 3600)
            .await;

        if !final_response.is_empty() {
            let assistant_turn = ConversationTurn {
                role: "assistant".to_string(),
                content: final_response.clone(),
                timestamp: now,
            };
            let _ = state
                .list_push_capped(&conv_key, &assistant_turn, 20, 3600)
                .await;
        }
    }

    Ok(final_response)
}
