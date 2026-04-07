use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use expert_types::activity::{ActivityLifecycle, ActivityState};
use expert_types::event_filter::EventFilter;
use expert_types::goal::Goal;
use expert_types::signals::{FireSignal, ToolDefinition};

/// In-memory registry of all activities. Persisted to Redis on mutation.
pub struct ActivityRegistry {
    pub activities: HashMap<String, ManagedActivity>,
}

pub struct ManagedActivity {
    pub state: ActivityState,
    pub goals: Vec<Goal>,
    pub tool_definitions: Vec<ToolDefinition>,
    pub event_filter: EventFilter,
    pub worker_id: String,
    /// Pending fire signal (depth-1 queue per activity).
    pub pending_fire: Option<PendingFire>,
}

pub struct PendingFire {
    pub signal: FireSignal,
    pub received_at: u64,
}

impl ActivityRegistry {
    pub fn new() -> Self {
        Self {
            activities: HashMap::new(),
        }
    }

    pub fn create_activity(
        &mut self,
        activity_id: String,
        stream_id: String,
        domain: String,
        goals: Vec<Goal>,
        tool_definitions: Vec<ToolDefinition>,
        event_filter: EventFilter,
        worker_id: String,
    ) -> &ManagedActivity {
        let now = now_ms();
        let k = goals.len();
        let dim = goals.first().map(|g| g.embedding.len()).unwrap_or(0);

        let goal_matrix: Vec<f32> = goals.iter().flat_map(|g| g.embedding.clone()).collect();
        let goal_indices: Vec<String> = goals.iter().map(|g| g.id.clone()).collect();

        let state = ActivityState {
            activity_id: activity_id.clone(),
            stream_id,
            domain,
            goal_tree_root_id: goals.first().map(|g| g.id.clone()).unwrap_or_default(),
            goal_matrix,
            goal_indices,
            h: vec![0.0; 256],   // SSM hidden dim
            theta: vec![0.5; k], // initial thresholds
            ema: 0.0,
            centroid: vec![0.0; dim],
            cov_matrix: vec![1.0; dim], // diagonal variance (all 1s initially)
            stream_mean: vec![0.0; dim],
            firing_history: Vec::new(),
            suppress_count: 0,
            recall_count: 0,
            invocation_count: 0,
            event_count: 0,
            pending_fire: None,
            refractory_until: 0,
            lifecycle_state: ActivityLifecycle::ColdStart,
            created_at: now,
            last_active: now,
            session_history_id: None,
            event_filter: event_filter.clone(),
        };

        let managed = ManagedActivity {
            state,
            goals,
            tool_definitions,
            event_filter,
            worker_id,
            pending_fire: None,
        };

        self.activities.insert(activity_id.clone(), managed);
        self.activities.get(&activity_id).unwrap()
    }

    pub fn get(&self, id: &str) -> Option<&ManagedActivity> {
        self.activities.get(id)
    }

    pub fn list(&self) -> Vec<&ManagedActivity> {
        self.activities.values().collect()
    }

    pub fn remove(&mut self, id: &str) -> Option<ManagedActivity> {
        self.activities.remove(id)
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use expert_types::goal::GoalAggregation;

    fn make_goal(name: &str, dim: usize) -> Goal {
        Goal {
            id: format!("goal-{name}"),
            name: name.to_string(),
            description: format!("Watch for {name}"),
            embedding: vec![0.1; dim],
            parent_id: None,
            children: Vec::new(),
            aggregation: GoalAggregation::Max,
            weights: None,
            domain: Some("test".to_string()),
            created_at: 0,
            version: 1,
            active: true,
        }
    }

    #[test]
    fn test_create_and_get() {
        let mut reg = ActivityRegistry::new();
        let goals = vec![make_goal("a", 4)];
        reg.create_activity(
            "act-1".into(),
            "stream-1".into(),
            "test".into(),
            goals,
            Vec::new(),
            EventFilter::All,
            "w-1".into(),
        );
        assert!(reg.get("act-1").is_some());
        assert!(reg.get("act-999").is_none());
    }

    #[test]
    fn test_list() {
        let mut reg = ActivityRegistry::new();
        assert_eq!(reg.list().len(), 0);

        reg.create_activity(
            "a1".into(),
            "s1".into(),
            "d".into(),
            vec![make_goal("x", 4)],
            Vec::new(),
            EventFilter::All,
            "w".into(),
        );
        reg.create_activity(
            "a2".into(),
            "s1".into(),
            "d".into(),
            vec![make_goal("y", 4)],
            Vec::new(),
            EventFilter::All,
            "w".into(),
        );
        assert_eq!(reg.list().len(), 2);
    }

    #[test]
    fn test_remove() {
        let mut reg = ActivityRegistry::new();
        reg.create_activity(
            "act-1".into(),
            "s1".into(),
            "d".into(),
            vec![make_goal("a", 4)],
            Vec::new(),
            EventFilter::All,
            "w".into(),
        );
        let removed = reg.remove("act-1");
        assert!(removed.is_some());
        assert!(reg.get("act-1").is_none());
        assert!(reg.remove("act-1").is_none());
    }

    #[test]
    fn test_goal_matrix_shape() {
        let mut reg = ActivityRegistry::new();
        let dim = 8;
        let goals = vec![
            make_goal("a", dim),
            make_goal("b", dim),
            make_goal("c", dim),
        ];
        let k = goals.len();
        reg.create_activity(
            "act-1".into(),
            "s1".into(),
            "d".into(),
            goals,
            Vec::new(),
            EventFilter::All,
            "w".into(),
        );
        let managed = reg.get("act-1").unwrap();
        assert_eq!(managed.state.goal_matrix.len(), k * dim);
        assert_eq!(managed.state.goal_indices.len(), k);
        assert_eq!(managed.state.theta.len(), k);
    }

    #[test]
    fn test_initial_lifecycle_state() {
        let mut reg = ActivityRegistry::new();
        reg.create_activity(
            "act-1".into(),
            "s1".into(),
            "d".into(),
            vec![make_goal("a", 4)],
            Vec::new(),
            EventFilter::All,
            "w".into(),
        );
        let managed = reg.get("act-1").unwrap();
        assert_eq!(managed.state.lifecycle_state, ActivityLifecycle::ColdStart);
    }
}
