use std::collections::VecDeque;

use serde::Serialize;

/// Sliding-window accumulator for pipeline timing metrics.
pub struct PipelineMetrics {
    max_window: usize,
    pub total_latency: VecDeque<f64>,
    pub context_assembly: VecDeque<f64>,
    pub llm_invocation: VecDeque<f64>,
    pub tokens_per_sec: VecDeque<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MetricSummary {
    pub min: f64,
    pub max: f64,
    pub avg: f64,
    pub p50: f64,
    pub p99: f64,
    pub count: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct MetricsSnapshot {
    pub total_latency: Option<MetricSummary>,
    pub context_assembly: Option<MetricSummary>,
    pub llm_invocation: Option<MetricSummary>,
    pub tokens_per_sec: Option<MetricSummary>,
    pub invocation_count: usize,
}

impl PipelineMetrics {
    pub fn new(window: usize) -> Self {
        Self {
            max_window: window,
            total_latency: VecDeque::new(),
            context_assembly: VecDeque::new(),
            llm_invocation: VecDeque::new(),
            tokens_per_sec: VecDeque::new(),
        }
    }

    pub fn record_total_latency(&mut self, ms: f64) {
        self.total_latency.push_back(ms);
        while self.total_latency.len() > self.max_window {
            self.total_latency.pop_front();
        }
    }

    pub fn record_context_assembly(&mut self, ms: f64) {
        self.context_assembly.push_back(ms);
        while self.context_assembly.len() > self.max_window {
            self.context_assembly.pop_front();
        }
    }

    pub fn record_llm_invocation(&mut self, ms: f64) {
        self.llm_invocation.push_back(ms);
        while self.llm_invocation.len() > self.max_window {
            self.llm_invocation.pop_front();
        }
    }

    pub fn record_tokens_per_sec(&mut self, tps: f64) {
        self.tokens_per_sec.push_back(tps);
        while self.tokens_per_sec.len() > self.max_window {
            self.tokens_per_sec.pop_front();
        }
    }

    pub fn snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            total_latency: summarize(&self.total_latency),
            context_assembly: summarize(&self.context_assembly),
            llm_invocation: summarize(&self.llm_invocation),
            tokens_per_sec: summarize(&self.tokens_per_sec),
            invocation_count: self.total_latency.len(),
        }
    }
}

fn summarize(data: &VecDeque<f64>) -> Option<MetricSummary> {
    if data.is_empty() {
        return None;
    }
    let mut sorted: Vec<f64> = data.iter().copied().collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    let sum: f64 = sorted.iter().sum();

    Some(MetricSummary {
        min: sorted[0],
        max: sorted[n - 1],
        avg: sum / n as f64,
        p50: percentile(&sorted, 50.0),
        p99: percentile(&sorted, 99.0),
        count: n,
    })
}

fn percentile(sorted: &[f64], pct: f64) -> f64 {
    if sorted.len() == 1 {
        return sorted[0];
    }
    let rank = pct / 100.0 * (sorted.len() - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    let frac = rank - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}
