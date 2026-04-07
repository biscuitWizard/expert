use expert_types::signals::ThresholdHint;

/// Configuration for adaptive threshold updates (spec Section 5.6).
pub struct ThresholdConfig {
    pub max_suppress_rate: f32,
    pub max_recall_rate: f32,
    pub raise_factor: f32,
    pub lower_factor: f32,
    pub hint_weight: f32,
    pub theta_min: f32,
    pub theta_max: f32,
}

impl Default for ThresholdConfig {
    fn default() -> Self {
        Self {
            max_suppress_rate: 0.5,
            max_recall_rate: 0.3,
            raise_factor: 1.05,
            lower_factor: 0.97,
            hint_weight: 0.02,
            theta_min: 0.1,
            theta_max: 0.95,
        }
    }
}

/// Update per-goal thresholds based on observed suppress/recall rates and LLM hints.
///
/// - If suppress rate exceeds max, thresholds are too low (firing too often) -> raise.
/// - If recall rate exceeds max, thresholds are too high (missing events) -> lower.
/// - LLM hints apply a weighted delta per goal.
pub fn update_thresholds(
    theta: &mut [f32],
    suppress_rate: f32,
    recall_rate: f32,
    hints: &[ThresholdHint],
    config: &ThresholdConfig,
) {
    for t in theta.iter_mut() {
        if suppress_rate > config.max_suppress_rate {
            *t *= config.raise_factor;
        } else if recall_rate > config.max_recall_rate {
            *t *= config.lower_factor;
        }
    }

    for hint in hints {
        if let Some(t) = theta.iter_mut().nth(0) {
            let delta = match hint.magnitude {
                expert_types::signals::ThresholdMagnitude::Slight => 0.01,
                expert_types::signals::ThresholdMagnitude::Moderate => 0.03,
                expert_types::signals::ThresholdMagnitude::Strong => 0.05,
            };
            let signed_delta = match hint.direction {
                expert_types::signals::ThresholdDirection::Raise => delta,
                expert_types::signals::ThresholdDirection::Lower => -delta,
            };
            *t += signed_delta * config.hint_weight;
        }
    }

    for t in theta.iter_mut() {
        *t = t.clamp(config.theta_min, config.theta_max);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use expert_types::signals::{ThresholdDirection, ThresholdMagnitude};

    fn default_config() -> ThresholdConfig {
        ThresholdConfig::default()
    }

    #[test]
    fn test_high_suppress_rate_raises_thresholds() {
        let config = default_config();
        let mut theta = vec![0.5, 0.6];
        update_thresholds(&mut theta, 0.7, 0.0, &[], &config);
        assert!(theta[0] > 0.5);
        assert!(theta[1] > 0.6);
    }

    #[test]
    fn test_high_recall_rate_lowers_thresholds() {
        let config = default_config();
        let mut theta = vec![0.5, 0.6];
        update_thresholds(&mut theta, 0.0, 0.5, &[], &config);
        assert!(theta[0] < 0.5);
        assert!(theta[1] < 0.6);
    }

    #[test]
    fn test_normal_rates_no_change() {
        let config = default_config();
        let mut theta = vec![0.5, 0.6];
        update_thresholds(&mut theta, 0.2, 0.1, &[], &config);
        assert!((theta[0] - 0.5).abs() < 1e-6);
        assert!((theta[1] - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_clipping_respects_bounds() {
        let config = default_config();
        let mut theta = vec![0.94];
        update_thresholds(&mut theta, 0.8, 0.0, &[], &config);
        assert!(theta[0] <= config.theta_max);

        let mut theta = vec![0.11];
        update_thresholds(&mut theta, 0.0, 0.5, &[], &config);
        assert!(theta[0] >= config.theta_min);
    }

    #[test]
    fn test_hint_applied() {
        let config = default_config();
        let mut theta = vec![0.5];
        let hints = vec![ThresholdHint {
            activity_id: "a".into(),
            goal_id: "g".into(),
            direction: ThresholdDirection::Raise,
            magnitude: ThresholdMagnitude::Strong,
        }];
        update_thresholds(&mut theta, 0.2, 0.1, &hints, &config);
        assert!(theta[0] > 0.5);
    }
}
