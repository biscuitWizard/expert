use ndarray::ArrayView1;
use tracing::debug;

use expert_config::Config;
use expert_types::activity::ActivityLifecycle;
use expert_types::event::Event;
use expert_types::goal::Goal;
use expert_types::signals::FireSignal;

use expert_ssm::features::{FeatureState, compute_features};
use expert_ssm::ssm::{LinearSsm, SsmCore};

const NUM_SCALAR_FEATURES: usize = 3; // drift, surprise, delta_t (silences are per-goal, handled dynamically)

pub struct ActivityInstance {
    pub activity_id: String,
    pub stream_id: String,
    pub goals: Vec<Goal>,
    pub ssm: LinearSsm,
    pub feature_state: FeatureState,
    pub theta: Vec<f32>,
    pub lifecycle: ActivityLifecycle,
    pub debounce_ms: u64,
    pub refractory_ms: u64,

    // Debounce state
    pending_fire_goals: Option<Vec<usize>>,
    pending_fire_scores: Option<Vec<f32>>,
    pending_fire_at: u64,
    pending_trigger_seq: Option<String>,

    // Refractory
    refractory_until: u64,

    // Tracking
    last_fired_seq: Option<String>,
    last_entry_id: Option<String>,
    event_count: u64,
}

impl ActivityInstance {
    pub fn new(activity_id: String, stream_id: String, goals: Vec<Goal>, config: &Config) -> Self {
        let k = goals.len();
        let embedding_dim = config.embedding_dim;
        let hidden_dim = config.ssm_hidden_dim;

        // input_dim for SSM: hidden_dim (projected) + k (cosines) + 3 scalars + k silences
        let max_k = k.max(16); // reserve space for future goals
        let ssm = LinearSsm::new(
            embedding_dim,
            hidden_dim,
            max_k,
            NUM_SCALAR_FEATURES + max_k, // scalars + silences
        );

        let feature_state = FeatureState::new(embedding_dim, k);

        Self {
            activity_id,
            stream_id,
            goals,
            ssm,
            feature_state,
            theta: vec![0.5; k],
            lifecycle: ActivityLifecycle::ColdStart,
            debounce_ms: config.debounce_ms,
            refractory_ms: config.refractory_ms,
            pending_fire_goals: None,
            pending_fire_scores: None,
            pending_fire_at: 0,
            pending_trigger_seq: None,
            refractory_until: 0,
            last_fired_seq: None,
            last_entry_id: None,
            event_count: 0,
        }
    }

    pub fn process_event(&mut self, event: &Event, now: u64, _config: &Config) {
        // Check refractory
        if self.lifecycle == ActivityLifecycle::Refractory {
            if now >= self.refractory_until {
                self.lifecycle = ActivityLifecycle::Active;
            } else {
                // Still in refractory: update hidden state but don't score
                if let Some(ref emb) = event.embedding {
                    let emb_view = ArrayView1::from(emb.as_slice());
                    let projected = self.ssm.project_embedding(emb_view);
                    let k = self.goals.len();
                    let goal_views: Vec<ArrayView1<f32>> = self
                        .goals
                        .iter()
                        .map(|g| ArrayView1::from(g.embedding.as_slice()))
                        .collect();
                    let features = compute_features(
                        emb_view,
                        &goal_views,
                        &mut self.feature_state,
                        event.timestamp,
                        &projected,
                    );
                    let _ = self.ssm.update(&features, k);
                }
                self.event_count += 1;
                return;
            }
        }

        // Transition from ColdStart to Active after first event
        if self.lifecycle == ActivityLifecycle::ColdStart {
            self.lifecycle = ActivityLifecycle::Active;
        }

        if self.lifecycle == ActivityLifecycle::Fired {
            // Activity is in Fired state (waiting for LLM invocation to complete)
            // Still update hidden state
            if let Some(ref emb) = event.embedding {
                let emb_view = ArrayView1::from(emb.as_slice());
                let projected = self.ssm.project_embedding(emb_view);
                let k = self.goals.len();
                let goal_views: Vec<ArrayView1<f32>> = self
                    .goals
                    .iter()
                    .map(|g| ArrayView1::from(g.embedding.as_slice()))
                    .collect();
                let features = compute_features(
                    emb_view,
                    &goal_views,
                    &mut self.feature_state,
                    event.timestamp,
                    &projected,
                );
                let _ = self.ssm.update(&features, k);
            }
            self.event_count += 1;
            return;
        }

        let embedding = match &event.embedding {
            Some(emb) => emb,
            None => return, // Skip events without embeddings
        };

        let emb_view = ArrayView1::from(embedding.as_slice());
        let projected = self.ssm.project_embedding(emb_view);
        let k = self.goals.len();

        let goal_views: Vec<ArrayView1<f32>> = self
            .goals
            .iter()
            .map(|g| ArrayView1::from(g.embedding.as_slice()))
            .collect();

        let features = compute_features(
            emb_view,
            &goal_views,
            &mut self.feature_state,
            event.timestamp,
            &projected,
        );

        let scores = self.ssm.update(&features, k);
        self.event_count += 1;

        // Track entry ID for XREVRANGE bounds
        // Use the event sequence as a proxy for the entry ID
        self.last_entry_id = Some(format!("{}", event.sequence));

        // Threshold check (only in Active state, not during pending debounce)
        if self.lifecycle == ActivityLifecycle::Active && self.pending_fire_goals.is_none() {
            let mut triggered = Vec::new();
            for (i, (&score, &threshold)) in scores.iter().zip(self.theta.iter()).enumerate() {
                if score >= threshold {
                    triggered.push(i);
                }
            }

            if !triggered.is_empty() {
                debug!(
                    activity_id = %self.activity_id,
                    triggered = ?triggered,
                    scores = ?scores,
                    "threshold crossed, entering debounce"
                );
                self.lifecycle = ActivityLifecycle::PendingFire;
                self.pending_fire_goals = Some(triggered);
                self.pending_fire_scores = Some(scores);
                self.pending_fire_at = now;
                self.pending_trigger_seq = self.last_entry_id.clone();
            }
        }
    }

    /// Check if debounce period has elapsed and scores still hold.
    pub fn check_debounce(&mut self, now: u64) -> Option<FireSignal> {
        if self.lifecycle != ActivityLifecycle::PendingFire {
            return None;
        }

        let _goals = self.pending_fire_goals.as_ref()?;
        let fire_at = self.pending_fire_at;

        if now < fire_at + self.debounce_ms {
            return None; // Still in debounce window
        }

        // Debounce elapsed -- re-check is implicit (we use the scores at trigger time for MVP)
        let scores = self.pending_fire_scores.take()?;
        let triggered_goals = self.pending_fire_goals.take()?;
        let trigger_seq = self.pending_trigger_seq.take().unwrap_or_default();

        let firing_goal_ids: Vec<String> = triggered_goals
            .iter()
            .filter_map(|&i| self.goals.get(i).map(|g| g.id.clone()))
            .collect();

        let signal = FireSignal {
            activity_id: self.activity_id.clone(),
            stream_id: self.stream_id.clone(),
            firing_goal_ids,
            scores,
            trigger_event_seq: trigger_seq,
            last_fired_seq: self.last_fired_seq.clone(),
            timestamp: now,
        };

        // Transition to refractory after fire
        self.lifecycle = ActivityLifecycle::Refractory;
        self.refractory_until = now + self.refractory_ms;
        self.last_fired_seq = self.last_entry_id.clone();

        Some(signal)
    }

    /// Take a fire signal if one is ready (called from the event processing loop).
    pub fn take_fire_signal(&mut self) -> Option<FireSignal> {
        None // Fire signals are produced via check_debounce
    }

    pub fn update_goals(&mut self, goals: Vec<Goal>) {
        let k = goals.len();
        self.goals = goals;
        self.theta.resize(k, 0.5);
        self.feature_state.resize_goals(k);
    }

    pub fn load_checkpoint(
        &mut self,
        ckpt: &expert_ssm::checkpoint::SsmCheckpoint,
    ) -> anyhow::Result<()> {
        self.ssm.load_checkpoint(ckpt)?;
        self.lifecycle = ActivityLifecycle::ColdStart;
        Ok(())
    }

    pub fn apply_threshold_feedback(&mut self, suppress_rate: f32, recall_rate: f32) {
        let config = expert_ssm::threshold::ThresholdConfig::default();
        expert_ssm::threshold::update_thresholds(
            &mut self.theta,
            suppress_rate,
            recall_rate,
            &[],
            &config,
        );
    }
}
