//! The throughput predictor calibrated from one real measurement, which
//! absorbs the per-token cost the per-module roofline does not model.

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
        let _ = (linear_module_microseconds, measured_tokens_per_second);
        todo!("learn the constant offset and mark calibrated")
    }

    #[must_use]
    pub fn predict_tokens_per_second(&self, linear_module_microseconds: f64) -> f64 {
        let _ = linear_module_microseconds;
        todo!("1e6 / (linear + overhead), zero when the sum is not positive")
    }
}
