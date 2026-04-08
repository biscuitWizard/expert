use std::collections::HashMap;

use anyhow::Result;
use qdrant_client::qdrant::{
    Condition, CreateCollectionBuilder, DeletePointsBuilder, Distance, Filter, PointStruct,
    PointsIdsList, ScrollPointsBuilder, SearchPointsBuilder, UpsertPointsBuilder,
    VectorParamsBuilder,
};
use qdrant_client::{Payload, Qdrant};
use serde_json::json;
use tracing::info;

use expert_config::Config;
use expert_types::context::{Episode, SelfKnowledgeNode};
use expert_types::goal::Goal;

const EPISODES_COLLECTION: &str = "episodes";
const GOALS_COLLECTION: &str = "goals";
const SELF_KNOWLEDGE_COLLECTION: &str = "self_knowledge";

#[derive(Clone)]
pub struct QdrantStore {
    client: Qdrant,
    _dim: u64,
}

impl QdrantStore {
    pub async fn new(config: &Config) -> Result<Self> {
        let client = Qdrant::from_url(&config.qdrant_url).build()?;
        let dim = config.embedding_dim as u64;

        // Create collections if they don't exist
        for name in [
            EPISODES_COLLECTION,
            GOALS_COLLECTION,
            SELF_KNOWLEDGE_COLLECTION,
        ] {
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

        Ok(Self { client, _dim: dim })
    }

    pub async fn insert_episode(&self, episode: &Episode) -> Result<()> {
        if episode.embedding.is_empty() {
            return Ok(());
        }

        // Store a condensed prompt: strip verbose tool definitions and
        // keep only identity, goals, trigger event, and recent activity.
        let condensed_prompt = condense_prompt(&episode.rendered_prompt);

        let payload: Payload = json!({
            "activity_id": episode.activity_id,
            "goal_id": episode.goal_id,
            "domain": episode.domain,
            "trigger_event_id": episode.trigger_event_id,
            "rendered_prompt": condensed_prompt,
            "response": episode.response,
            "was_suppressed": episode.was_suppressed,
            "operator_forced": episode.operator_forced,
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
                embedding: Vec::new(),
                trigger_event_id: get_str("trigger_event_id"),
                trigger_scores: Vec::new(),
                rendered_prompt: get_str("rendered_prompt"),
                response: get_str("response"),
                tool_calls: Vec::new(),
                was_suppressed: get_bool("was_suppressed"),
                recalled_event_indices: Vec::new(),
                created_at: get_i64("created_at"),
                operator_forced: get_bool("operator_forced"),
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

    pub async fn upsert_self_knowledge(&self, node: &SelfKnowledgeNode) -> Result<()> {
        if node.embedding.is_empty() {
            return Ok(());
        }

        let payload: Payload = json!({
            "category": node.category,
            "content": node.content,
            "created_at": node.created_at,
            "updated_at": node.updated_at,
        })
        .try_into()
        .unwrap();

        let point = PointStruct::new(node.id.clone(), node.embedding.clone(), payload);

        self.client
            .upsert_points(UpsertPointsBuilder::new(
                SELF_KNOWLEDGE_COLLECTION,
                vec![point],
            ))
            .await?;

        Ok(())
    }

    pub async fn search_self_knowledge(
        &self,
        embedding: &[f32],
        k: usize,
    ) -> Result<Vec<SelfKnowledgeNode>> {
        if embedding.is_empty() {
            return Ok(Vec::new());
        }

        let results = self
            .client
            .search_points(
                SearchPointsBuilder::new(SELF_KNOWLEDGE_COLLECTION, embedding.to_vec(), k as u64)
                    .with_payload(true),
            )
            .await?;

        Ok(results
            .result
            .into_iter()
            .map(|p| point_to_node(&p))
            .collect())
    }

    /// Always returns the core_identity node if one exists, regardless of embedding similarity.
    pub async fn get_core_identity(&self) -> Result<Option<SelfKnowledgeNode>> {
        let filter = Filter::must([Condition::matches("category", "core_identity".to_string())]);

        let results = self
            .client
            .scroll(
                ScrollPointsBuilder::new(SELF_KNOWLEDGE_COLLECTION)
                    .filter(filter)
                    .limit(1)
                    .with_payload(true),
            )
            .await?;

        Ok(results
            .result
            .first()
            .map(|p| point_to_node_from_retrieved(p)))
    }

    /// Retrieve all active goals matching a domain.
    pub async fn get_goals_by_domain(&self, domain: &str) -> Result<Vec<Goal>> {
        let filter = Filter::must([
            Condition::matches("domain", domain.to_string()),
            Condition::matches("active", true),
        ]);

        let results = self
            .client
            .scroll(
                ScrollPointsBuilder::new(GOALS_COLLECTION)
                    .filter(filter)
                    .limit(100)
                    .with_payload(true)
                    .with_vectors(true),
            )
            .await?;

        let mut goals = Vec::new();
        for point in results.result {
            let payload = &point.payload;

            let get_str = |key: &str| -> String {
                payload
                    .get(key)
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .unwrap_or_default()
            };
            let get_i64 = |key: &str| -> u64 {
                payload
                    .get(key)
                    .and_then(|v| v.as_integer())
                    .map(|i| i as u64)
                    .unwrap_or(0)
            };
            let get_bool =
                |key: &str| -> bool { payload.get(key).and_then(|v| v.as_bool()).unwrap_or(false) };

            let id = match point.id {
                Some(ref pid) => match pid.point_id_options {
                    Some(qdrant_client::qdrant::point_id::PointIdOptions::Uuid(ref u)) => u.clone(),
                    _ => continue,
                },
                None => continue,
            };

            let embedding: Vec<f32> = point
                .vectors
                .and_then(|v| match v.vectors_options {
                    Some(qdrant_client::qdrant::vectors_output::VectorsOptions::Vector(vec)) => {
                        Some(vec.data.clone())
                    }
                    _ => None,
                })
                .unwrap_or_default();

            let domain_val = get_str("domain");

            goals.push(Goal {
                id,
                name: get_str("name"),
                description: get_str("description"),
                embedding,
                parent_id: None,
                children: Vec::new(),
                aggregation: expert_types::goal::GoalAggregation::Max,
                weights: None,
                domain: if domain_val.is_empty() {
                    None
                } else {
                    Some(domain_val)
                },
                created_at: get_i64("created_at"),
                version: get_i64("version") as u32,
                active: get_bool("active"),
            });
        }

        Ok(goals)
    }

    /// Deduplicate goals: for each (name, description, domain) group, keep only
    /// the most recent point and delete the rest.
    pub async fn dedup_goals(&self) -> Result<usize> {
        let results = self
            .client
            .scroll(
                ScrollPointsBuilder::new(GOALS_COLLECTION)
                    .limit(500)
                    .with_payload(true),
            )
            .await?;

        // Group by (name, description, domain)
        let mut groups: HashMap<(String, String, String), Vec<(String, u64)>> = HashMap::new();
        for point in &results.result {
            let payload = &point.payload;
            let get_str = |key: &str| -> String {
                payload
                    .get(key)
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .unwrap_or_default()
            };
            let created_at = payload
                .get("created_at")
                .and_then(|v| v.as_integer())
                .map(|i| i as u64)
                .unwrap_or(0);

            let id = match point.id {
                Some(ref pid) => match pid.point_id_options {
                    Some(qdrant_client::qdrant::point_id::PointIdOptions::Uuid(ref u)) => u.clone(),
                    _ => continue,
                },
                None => continue,
            };

            let key = (get_str("name"), get_str("description"), get_str("domain"));
            groups.entry(key).or_default().push((id, created_at));
        }

        let mut to_delete = Vec::new();
        for (_, mut entries) in groups {
            if entries.len() <= 1 {
                continue;
            }
            entries.sort_by(|a, b| b.1.cmp(&a.1));
            for (id, _) in entries.into_iter().skip(1) {
                to_delete.push(id);
            }
        }

        let deleted = to_delete.len();
        if !to_delete.is_empty() {
            let point_ids: Vec<qdrant_client::qdrant::PointId> = to_delete
                .into_iter()
                .map(|id| qdrant_client::qdrant::PointId::from(id))
                .collect();

            self.client
                .delete_points(
                    DeletePointsBuilder::new(GOALS_COLLECTION)
                        .points(PointsIdsList { ids: point_ids }),
                )
                .await?;

            info!(deleted, "deduplicated goals collection");
        }

        Ok(deleted)
    }

    /// Seed the core identity node if it doesn't already exist.
    /// Inserted with an empty embedding; the rag-service will backfill
    /// the embedding on first `get_self_knowledge` query.
    pub async fn seed_identity(&self, content: &str) -> Result<bool> {
        if self.get_core_identity().await?.is_some() {
            return Ok(false);
        }

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        let id = uuid::Uuid::new_v4().to_string();
        let payload: Payload = json!({
            "category": "core_identity",
            "content": content,
            "created_at": now,
            "updated_at": now,
        })
        .try_into()
        .unwrap();

        let dim = self._dim as usize;
        let zero_vec = vec![0.0f32; dim];
        let point = PointStruct::new(id, zero_vec, payload);

        self.client
            .upsert_points(UpsertPointsBuilder::new(
                SELF_KNOWLEDGE_COLLECTION,
                vec![point],
            ))
            .await?;

        Ok(true)
    }
}

fn point_to_node(point: &qdrant_client::qdrant::ScoredPoint) -> SelfKnowledgeNode {
    let payload = &point.payload;

    let get_str = |key: &str| -> String {
        payload
            .get(key)
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_default()
    };

    let get_u64 = |key: &str| -> u64 {
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

    SelfKnowledgeNode {
        id,
        category: get_str("category"),
        content: get_str("content"),
        embedding: Vec::new(),
        created_at: get_u64("created_at"),
        updated_at: get_u64("updated_at"),
    }
}

fn point_to_node_from_retrieved(
    point: &qdrant_client::qdrant::RetrievedPoint,
) -> SelfKnowledgeNode {
    let payload = &point.payload;

    let get_str = |key: &str| -> String {
        payload
            .get(key)
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_default()
    };

    let get_u64 = |key: &str| -> u64 {
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

    SelfKnowledgeNode {
        id,
        category: get_str("category"),
        content: get_str("content"),
        embedding: Vec::new(),
        created_at: get_u64("created_at"),
        updated_at: get_u64("updated_at"),
    }
}

/// Strip verbose sections (tool definitions, long tool lists) from the
/// rendered prompt before persisting to Qdrant. Keeps identity, goals,
/// trigger event, recent activity, and conversation history.
fn condense_prompt(prompt: &str) -> String {
    let mut result = String::with_capacity(prompt.len());
    let mut skip = false;

    for line in prompt.lines() {
        if line.starts_with("=== AVAILABLE TOOLS ===") {
            skip = true;
            result.push_str("[tools omitted]\n");
            continue;
        }
        if skip && line.starts_with("=== ") {
            skip = false;
        }
        if !skip {
            result.push_str(line);
            result.push('\n');
        }
    }

    result.truncate(result.trim_end().len());
    result
}
