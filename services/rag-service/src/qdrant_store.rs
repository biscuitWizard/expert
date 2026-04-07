use anyhow::Result;
use qdrant_client::qdrant::{
    CreateCollectionBuilder, Distance, PointStruct, SearchPointsBuilder, UpsertPointsBuilder,
    VectorParamsBuilder,
};
use qdrant_client::{Payload, Qdrant};
use serde_json::json;
use tracing::info;

use expert_config::Config;
use expert_types::context::Episode;
use expert_types::goal::Goal;

const EPISODES_COLLECTION: &str = "episodes";
const GOALS_COLLECTION: &str = "goals";

#[derive(Clone)]
pub struct QdrantStore {
    client: Qdrant,
    dim: u64,
}

impl QdrantStore {
    pub async fn new(config: &Config) -> Result<Self> {
        let client = Qdrant::from_url(&config.qdrant_url).build()?;
        let dim = config.embedding_dim as u64;

        // Create collections if they don't exist
        for name in [EPISODES_COLLECTION, GOALS_COLLECTION] {
            if !client.collection_exists(name).await? {
                client
                    .create_collection(
                        CreateCollectionBuilder::new(name)
                            .vectors_config(VectorParamsBuilder::new(dim, Distance::Cosine)),
                    )
                    .await?;
                info!(collection = name, dim, "created qdrant collection");
            }
        }

        Ok(Self { client, dim })
    }

    pub async fn insert_episode(&self, episode: &Episode) -> Result<()> {
        if episode.embedding.is_empty() {
            return Ok(());
        }

        let payload: Payload = json!({
            "activity_id": episode.activity_id,
            "goal_id": episode.goal_id,
            "domain": episode.domain,
            "trigger_event_id": episode.trigger_event_id,
            "rendered_prompt": episode.rendered_prompt,
            "response": episode.response,
            "was_suppressed": episode.was_suppressed,
            "created_at": episode.created_at,
        })
        .try_into()
        .unwrap();

        let point = PointStruct::new(episode.id.clone(), episode.embedding.clone(), payload);

        self.client
            .upsert_points(UpsertPointsBuilder::new(EPISODES_COLLECTION, vec![point]))
            .await?;

        Ok(())
    }

    pub async fn search_episodes(&self, embedding: &[f32], k: usize) -> Result<Vec<Episode>> {
        if embedding.is_empty() {
            return Ok(Vec::new());
        }

        let results = self
            .client
            .search_points(
                SearchPointsBuilder::new(EPISODES_COLLECTION, embedding.to_vec(), k as u64)
                    .with_payload(true),
            )
            .await?;

        let mut episodes = Vec::new();
        for point in results.result {
            let payload = &point.payload;

            let get_str = |key: &str| -> String {
                payload
                    .get(key)
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .unwrap_or_default()
            };

            let get_bool =
                |key: &str| -> bool { payload.get(key).and_then(|v| v.as_bool()).unwrap_or(false) };

            let get_i64 = |key: &str| -> u64 {
                payload
                    .get(key)
                    .and_then(|v| v.as_integer())
                    .map(|i| i as u64)
                    .unwrap_or(0)
            };

            let id = match point.id {
                Some(ref pid) => match pid.point_id_options {
                    Some(qdrant_client::qdrant::point_id::PointIdOptions::Uuid(ref u)) => u.clone(),
                    _ => String::new(),
                },
                None => String::new(),
            };

            episodes.push(Episode {
                id,
                activity_id: get_str("activity_id"),
                goal_id: get_str("goal_id"),
                domain: {
                    let d = get_str("domain");
                    if d.is_empty() { None } else { Some(d) }
                },
                embedding: Vec::new(), // Don't return embeddings in search results
                trigger_event_id: get_str("trigger_event_id"),
                trigger_scores: Vec::new(),
                rendered_prompt: get_str("rendered_prompt"),
                response: get_str("response"),
                tool_calls: Vec::new(),
                was_suppressed: get_bool("was_suppressed"),
                recalled_event_indices: Vec::new(),
                created_at: get_i64("created_at"),
            });
        }

        Ok(episodes)
    }

    pub async fn upsert_goal(&self, goal: &Goal) -> Result<()> {
        if goal.embedding.is_empty() {
            return Ok(());
        }

        let payload: Payload = json!({
            "name": goal.name,
            "description": goal.description,
            "activity_id": "",
            "domain": goal.domain,
            "version": goal.version,
            "active": goal.active,
            "created_at": goal.created_at,
        })
        .try_into()
        .unwrap();

        let point = PointStruct::new(goal.id.clone(), goal.embedding.clone(), payload);

        self.client
            .upsert_points(UpsertPointsBuilder::new(GOALS_COLLECTION, vec![point]))
            .await?;

        Ok(())
    }
}
