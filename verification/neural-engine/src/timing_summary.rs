//! Order statistics over repeated timings of one operation.

use std::time::Duration;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TimingSummary {
    pub sample_count: usize,
    pub minimum: Duration,
    pub median: Duration,
    pub ninetieth_percentile: Duration,
    pub maximum: Duration,
}

impl TimingSummary {
    /// Summarizes `samples`; `None` when there are none.
    ///
    /// Percentiles use the nearest-rank method, so every reported value is a
    /// time that was actually observed rather than an interpolation.
    #[must_use]
    pub fn from_samples(samples: &[Duration]) -> Option<Self> {
        if samples.is_empty() {
            return None;
        }
        let mut ordered_samples = samples.to_vec();
        ordered_samples.sort_unstable();
        Some(Self {
            sample_count: ordered_samples.len(),
            minimum: ordered_samples[0],
            median: ordered_samples[nearest_rank_index(ordered_samples.len(), 50)],
            ninetieth_percentile: ordered_samples[nearest_rank_index(ordered_samples.len(), 90)],
            maximum: ordered_samples[ordered_samples.len() - 1],
        })
    }

    /// Achieved throughput at the median time.
    #[must_use]
    pub fn median_teraflops(&self, floating_point_operations: f64) -> f64 {
        floating_point_operations / self.median.as_secs_f64() / 1.0e12
    }
}

fn nearest_rank_index(sample_count: usize, percentile: usize) -> usize {
    let rank = (percentile * sample_count).div_ceil(100);
    rank.max(1) - 1
}

#[cfg(test)]
#[path = "timing_summary_tests.rs"]
mod tests;
