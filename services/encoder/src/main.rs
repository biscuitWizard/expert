use std::time::Duration;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio::time::timeout;
use tracing::{error, info, warn};

use expert_config::Config;
use expert_redis::names;
use expert_redis::{StreamConsumer, StreamProducer};
use expert_types::event::Event;
use expert_types::signals::{EncodeRequest, EncodeResult};

// ---------- Ollama embedding client (OpenAI-compatible endpoint) ----------

#[derive(Serialize)]
struct EmbeddingsRequest {
    model: String,
    input: Vec<String>,
}

#[derive(Deserialize)]
struct EmbeddingsResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

struct OllamaEmbedder {
    client: reqwest::Client,
    url: String,
    model: String,
}

impl OllamaEmbedder {
    fn new(base_url: &str, model: &str) -> Self {
        Self {
            client: reqwest::Client::new(),
            url: format!("{}/v1/embeddings", base_url.trim_end_matches('/')),
            model: model.to_string(),
        }
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let req = EmbeddingsRequest {
            model: self.model.clone(),
            input: texts.to_vec(),
        };
        let resp = self
            .client
            .post(&self.url)
            .json(&req)
            .send()
            .await?
            .error_for_status()?;

        let result: EmbeddingsResponse = resp.json().await?;
        Ok(result.data.into_iter().map(|d| d.embedding).collect())
    }
}

// ---------- batch items ----------

enum BatchItem {
    RawEvent { event: Event },
    EncodeReq { req: EncodeRequest },
}

#[tokio::main]
async fn main() -> Result<()> {
    expert_config::init_tracing();
    let config = Config::from_env();
    info!("starting encoder service");

    let model = OllamaEmbedder::new(&config.ollama_embeddings_url, &config.embeddings_model);
    let conn = expert_redis::connect(&config.redis_url).await?;
    let mut producer = StreamProducer::new(conn.clone(), config.stream_maxlen);

    let (batch_tx, mut batch_rx) = mpsc::channel::<BatchItem>(256);

    // Spawn raw event consumer (wildcard via pattern subscription)
    // For MVP: consume from a known stream pattern. The consumer reads from
    // `events.raw.*` by subscribing to the group on each stream the system knows about.
    // Simplified: we use a configurable list or a single known stream.
    {
        let tx = batch_tx.clone();
        let conn = conn.clone();
        let stream_id = std::env::var("STREAM_ID").unwrap_or_else(|_| "stdin-01".to_string());
        tokio::spawn(async move {
            let stream = names::events_raw(&stream_id);
            let mut consumer = match StreamConsumer::new(
                conn,
                stream,
                "encoder".to_string(),
                "encoder-0".to_string(),
                200,
            )
            .await
            {
                Ok(c) => c,
                Err(e) => {
                    error!(error = %e, "failed to create raw event consumer");
                    return;
                }
            };

            loop {
                match consumer.consume::<Event>().await {
                    Ok(Some((id, event))) => {
                        if tx.send(BatchItem::RawEvent { event }).await.is_err() {
                            break;
                        }
                        let _ = consumer.ack(&id).await;
                    }
                    Ok(None) => {}
                    Err(e) => {
                        warn!(error = %e, "raw event consumer error");
                        tokio::time::sleep(Duration::from_secs(1)).await;
                    }
                }
            }
        });
    }

    // Spawn on-demand encode request consumer
    {
        let tx = batch_tx.clone();
        let conn = conn.clone();
        tokio::spawn(async move {
            let mut consumer = match StreamConsumer::new(
                conn,
                names::REQUESTS_ENCODE.to_string(),
                "encoder-od".to_string(),
                "encoder-od-0".to_string(),
                200,
            )
            .await
            {
                Ok(c) => c,
                Err(e) => {
                    error!(error = %e, "failed to create encode request consumer");
                    return;
                }
            };

            loop {
                match consumer.consume::<EncodeRequest>().await {
                    Ok(Some((id, req))) => {
                        if tx.send(BatchItem::EncodeReq { req }).await.is_err() {
                            break;
                        }
                        let _ = consumer.ack(&id).await;
                    }
                    Ok(None) => {}
                    Err(e) => {
                        warn!(error = %e, "encode request consumer error");
                        tokio::time::sleep(Duration::from_secs(1)).await;
                    }
                }
            }
        });
    }

    drop(batch_tx); // drop sender so batch loop can detect channel close

    // Micro-batching loop
    let max_batch = config.encoder_max_batch;
    let flush_timeout = Duration::from_millis(config.encoder_flush_ms);

    info!(
        max_batch,
        flush_ms = config.encoder_flush_ms,
        "micro-batcher started"
    );

    loop {
        let mut batch: Vec<BatchItem> = Vec::with_capacity(max_batch);

        // Wait for the first item (blocking)
        match batch_rx.recv().await {
            Some(item) => batch.push(item),
            None => {
                info!("batch channel closed, shutting down");
                break;
            }
        }

        // Try to fill up to max_batch or flush_timeout
        let deadline = tokio::time::Instant::now() + flush_timeout;
        while batch.len() < max_batch {
            match timeout(
                deadline.saturating_duration_since(tokio::time::Instant::now()),
                batch_rx.recv(),
            )
            .await
            {
                Ok(Some(item)) => batch.push(item),
                Ok(None) => break,
                Err(_) => break, // timeout
            }
        }

        // Extract texts for embedding
        let texts: Vec<String> = batch
            .iter()
            .map(|item| match item {
                BatchItem::RawEvent { event, .. } => event.raw.clone(),
                BatchItem::EncodeReq { req, .. } => req.text.clone(),
            })
            .collect();

        let embeddings = match model.embed(&texts).await {
            Ok(embs) => embs,
            Err(e) => {
                error!(error = %e, batch_size = texts.len(), "embedding request failed, dropping batch");
                continue;
            }
        };

        if embeddings.len() != batch.len() {
            error!(
                expected = batch.len(),
                got = embeddings.len(),
                "embedding count mismatch, dropping batch"
            );
            continue;
        }

        // Publish results
        for (item, embedding) in batch.into_iter().zip(embeddings.into_iter()) {
            match item {
                BatchItem::RawEvent { mut event } => {
                    event.embedding = Some(embedding);
                    let stream = names::events_embedded(&event.stream_id);
                    if let Err(e) = producer.publish(&stream, &event).await {
                        error!(error = %e, event_id = %event.id, "failed to publish embedded event");
                    }
                }
                BatchItem::EncodeReq { req } => {
                    let result = EncodeResult {
                        request_id: req.request_id,
                        embedding,
                    };
                    if let Err(e) = producer.publish(names::RESULTS_ENCODE, &result).await {
                        error!(error = %e, "failed to publish encode result");
                    }
                }
            }
        }
    }

    Ok(())
}
