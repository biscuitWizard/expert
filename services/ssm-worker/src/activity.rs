use std::collections::VecDeque;

use ndarray::ArrayView1;
use tracing::{debug, info};

use expert_config::Config;
use expert_types::activity::ActivityLifecycle;
use expert_types::event::Event;
use expert_types::event_filter::EventFilter;
use expert_types::goal::Goal;
use expert_types::signals::FireSignal;

use expert_ssm::features::{FeatureState, compute_features};
use expert_ssm::ssm::{LinearSsm, SsmCore};

const NUM_SCALAR_FEATURES: usize = 3; // drift, surprise, delta_t (silences are per-goal, handled dynamically)

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScoringMode {
    ColdCosine,
    SsmActive,
}

pub struct ActivityInstance {
    pub activity_id: String,
    pub stream_id: String,
    pub event_filter: EventFilter,
    pub goals: Vec<Goal>,
    pub ssm: LinearSsm,
    pub feature_state: FeatureState,
    pub theta: Vec<f32>,
    pub lifecycle: ActivityLifecycle,
    pub debounce_ms: u64,
    pub refractory_ms: u64,
    pub refractory_dm_ms: u64,
    initial_theta: f32,

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

    // Cold-start scoring
    pub scoring_mode: ScoringMode,
    pub fire_count: u64,
    cold_start_dm_fires: u64,
    calibration_window: usize,
    calibration_threshold: f32,
    cosine_history: VecDeque<Vec<f32>>,
    ssm_score_history: VecDeque<Vec<f32>>,
}

impl ActivityInstance {
    pub fn new(
        activity_id: String,
        stream_id: String,
        event_filter: EventFilter,
        goals: Vec<Goal>,
        config: &Config,
    ) -> Self {
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
            event_filter,
            goals,
            ssm,
            feature_state,
            theta: vec![config.ssm_initial_theta; k],
            lifecycle: ActivityLifecycle::ColdStart,
            debounce_ms: config.debounce_ms,
            refractory_ms: config.refractory_ms,
            refractory_dm_ms: config.refractory_dm_ms,
            initial_theta: config.ssm_initial_theta,
            pending_fire_goals: None,
            pending_fire_scores: None,
            pending_fire_at: 0,
            pending_trigger_seq: None,
            refractory_until: 0,
            last_fired_seq: None,
            last_entry_id: None,
            event_count: 0,
            scoring_mode: ScoringMode::ColdCosine,
            fire_count: 0,
            cold_start_dm_fires: config.cold_start_dm_fires,
            calibration_window: config.ssm_calibration_window,
            calibration_threshold: config.ssm_calibration_threshold,
            cosine_history: VecDeque::new(),
            ssm_score_history: VecDeque::new(),
        }
    }

    pub fn process_event(&mut self, event: &Event, entry_id: &str, now: u64, _config: &Config) {
        let is_dm = event.metadata.get("event_type").and_then(|v| v.as_str()) == Some("dm");

        // Check refractory -- DMs use a shorter window
        if self.lifecycle == ActivityLifecycle::Refractory {
            if now >= self.refractory_until {
                self.lifecycle = ActivityLifecycle::Active;
            } else if is_dm
                && now
                    >= self.refractory_until.saturating_sub(self.refractory_ms)
                        + self.refractory_dm_ms
            {
                self.lifecycle = ActivityLifecycle::Active;
            } else {
                self.update_hidden_state(event, now);
                self.event_count += 1;
                return;
            }
        }

        // ColdStart: transition to Active (first event only)
        if self.lifecycle == ActivityLifecycle::ColdStart {
            self.lifecycle = ActivityLifecycle::Active;
        }

        if self.lifecycle == ActivityLifecycle::Fired {
            self.update_hidden_state(event, now);
            self.event_count += 1;
            return;
        }

        let embedding = match &event.embedding {
            Some(emb) => emb,
            None => return,
        };

        let emb_view = ArrayView1::from(embedding.as_slice());
        let projected = self.ssm.project_embedding(emb_view);
        let hidden_dim = projected.len();
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

        // Extract cosine similarities from the feature vector
        let cosine_scores: Vec<f32> = features[hidden_dim..hidden_dim + k].to_vec();

        // Always run SSM to keep hidden state warm
        let ssm_scores = self.ssm.update(&features, k);
        self.event_count += 1;
        self.last_entry_id = Some(entry_id.to_string());

        // Track score history for calibration
        if self.scoring_mode == ScoringMode::ColdCosine {
            self.cosine_history.push_back(cosine_scores.clone());
            self.ssm_score_history.push_back(ssm_scores.clone());
            while self.cosine_history.len() > self.calibration_window {
                self.cosine_history.pop_front();
                self.ssm_score_history.pop_front();
            }
            self.check_ssm_calibration();
        }

        // Choose scores based on scoring mode
        let scores = match self.scoring_mode {
            ScoringMode::ColdCosine => cosine_scores,
            ScoringMode::SsmActive => ssm_scores,
        };

        // DM bypass during cold start: force-fire all DMs until we have enough data
        if is_dm
            && self.fire_count < self.cold_start_dm_fires
            && self.lifecycle == ActivityLifecycle::Active
            && self.pending_fire_goals.is_none()
        {
            let all_goals: Vec<usize> = (0..k).collect();
            debug!(
                activity_id = %self.activity_id,
                fire_count = self.fire_count,
                "cold-start DM bypass -- forcing fire"
            );
            self.lifecycle = ActivityLifecycle::PendingFire;
            self.pending_fire_goals = Some(all_goals);
            self.pending_fire_scores = Some(scores);
            self.pending_fire_at = now;
            self.pending_trigger_seq = self.last_entry_id.clone();
            return;
        }

        // Normal threshold check
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
                    mode = ?self.scoring_mode,
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

    fn update_hidden_state(&mut self, event: &Event, now: u64) {
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
                if now > 0 {
                    event.timestamp
                } else {
                    event.timestamp
                },
                &projected,
            );
            let _ = self.ssm.update(&features, k);
        }
    }

    fn check_ssm_calibration(&mut self) {
        if self.cosine_history.len() < self.calibration_window {
            return;
        }

        let k = self.goals.len();
        if k == 0 {
            return;
        }

        // Compute per-goal Pearson correlation between cosine and SSM scores
        let n = self.cosine_history.len() as f32;
        let mut total_corr = 0.0f32;
        let mut valid_goals = 0;

        for goal_idx in 0..k {
            let mut sum_x = 0.0f32;
            let mut sum_y = 0.0f32;
            let mut sum_xy = 0.0f32;
            let mut sum_x2 = 0.0f32;
            let mut sum_y2 = 0.0f32;

            for (cos_scores, ssm_scores) in self
                .cosine_history
                .iter()
                .zip(self.ssm_score_history.iter())
            {
                let x = cos_scores.get(goal_idx).copied().unwrap_or(0.0);
                let y = ssm_scores.get(goal_idx).copied().unwrap_or(0.0);
                sum_x += x;
                sum_y += y;
                sum_xy += x * y;
                sum_x2 += x * x;
                sum_y2 += y * y;
            }

            let denom = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();
            if denom > 1e-8 {
                let corr = (n * sum_xy - sum_x * sum_y) / denom;
                total_corr += corr;
                valid_goals += 1;
            }
        }

        if valid_goals > 0 {
            let avg_corr = total_corr / valid_goals as f32;
            if avg_corr >= self.calibration_threshold {
                info!(
                    activity_id = %self.activity_id,
                    correlation = avg_corr,
                    window = self.calibration_window,
                    "SSM calibrated -- transitioning to SSM scoring"
                );
                self.scoring_mode = ScoringMode::SsmActive;
                self.cosine_history.clear();
                self.ssm_score_history.clear();
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
            operator_forced: false,
        };

        // Transition to refractory after fire
        self.lifecycle = ActivityLifecycle::Refractory;
        self.refractory_until = now + self.refractory_ms;
        self.last_fired_seq = self.last_entry_id.clone();
        self.fire_count += 1;

        Some(signal)
    }

    /// Take a fire signal if one is ready (called from the event processing loop).
    pub fn take_fire_signal(&mut self) -> Option<FireSignal> {
        None // Fire signals are produced via check_debounce
    }

    pub fn update_goals(&mut self, goals: Vec<Goal>) {
        let k = goals.len();
        self.goals = goals;
        self.theta.resize(k, self.initial_theta);
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
