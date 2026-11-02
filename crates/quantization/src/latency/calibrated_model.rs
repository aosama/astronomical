//! The throughput predictor calibrated from one real measurement, which
//! absorbs the per-token cost the per-module roofline does not model.
//!
//! Ported from OptiQ's `CalibratedLatencyModel` (`optiq/core/latency.py`).

use super::hardware_profile::HardwareProfile;

/// Calibrates the model's constant per-token overhead (attention, norms,
/// cache, framework) from one real throughput measurement, after which it
/// predicts relative throughput between quantization configurations well.
#[derive(Clone, Debug, PartialEq)]
pub struct CalibratedLatencyModel {
    pub hardware: HardwareProfile,
    pub overhead_microseconds_per_token: f64,
    pub is_calibrated: bool,
}

impl CalibratedLatencyModel {
    /// `overhead = max(1e6 / measured_tokens_per_second - linear_microseconds, 0)`.
    pub fn calibrate(&mut self, linear_module_microseconds: f64, measured_tokens_per_second: f64) {
        let measured_microseconds_per_token = 1e6 / measured_tokens_per_second;
        self.overhead_microseconds_per_token =
            (measured_microseconds_per_token - linear_module_microseconds).max(0.0);
        self.is_calibrated = true;
    }

    /// `1e6 / (linear + overhead)`, zero when the sum is not positive.
    #[must_use]
    pub fn predict_tokens_per_second(&self, linear_module_microseconds: f64) -> f64 {
        let total_microseconds = linear_module_microseconds + self.overhead_microseconds_per_token;
        if total_microseconds > 0.0 {
            1e6 / total_microseconds
        } else {
            0.0
        }
    }
}
