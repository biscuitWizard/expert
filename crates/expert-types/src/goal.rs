use serde::{Deserialize, Serialize};

/// How a parent goal aggregates its children's scores.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GoalAggregation {
    Max,
    WeightedSum,
    All,
}

/// A semantic objective within an activity's goal tree.
///
/// Goals are atomic and per-activity: they are not shared between activities.
/// The orchestrator can copy goals across activities as a control-plane
/// operation, but each copy is independent.
///
/// Leaf goals have an embedding and appear as rows in the goal matrix G.
/// Internal (parent) goals derive their score by aggregating children.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Goal {
    pub id: String,
    pub name: String,
    pub description: String,
    /// Encoded from `description` by the encoder service. Dimension d.
    pub embedding: Vec<f32>,
    /// `None` for root goals.
    pub parent_id: Option<String>,
    pub children: Vec<String>,
    pub aggregation: GoalAggregation,
    /// Used when `aggregation` is `WeightedSum`. Length must match `children`.
    pub weights: Option<Vec<f32>>,
    /// Optional domain affiliation (e.g. "mud", "market", "logs").
    pub domain: Option<String>,
    /// Unix milliseconds.
    pub created_at: u64,
    /// Incremented on every update. Training data is tagged with the
    /// goal version active at time of generation (spec invariant 8).
    pub version: u32,
    pub active: bool,
}
