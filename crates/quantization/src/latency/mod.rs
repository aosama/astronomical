//! Hardware-aware latency model for Apple Silicon MLX inference.
//!
//! Models per-module inference cost with the roofline model: decode
//! (sequence length one) is memory-bandwidth-bound, so latency scales with
//! bytes read; prefill is compute-bound, so latency scales with floating
//! point operations over peak throughput. Lower bit widths read fewer bytes
//! per weight and so run faster at decode, but not linearly: per-group
//! scale and bias overhead, fixed dequantization cost per element, and
//! per-launch dispatch overhead all intrude.
//!
//! The port keeps OptiQ's analytical form and its per-width factors. Two
//! things change: the chip table is data the caller supplies (a published
//! specification table, or a profile measured on the host), never a
//! hardwired assumption about the machine running the code, and the
//! calibration against a real throughput measurement is the preferred path
//! because Astronomical already measures tokens per second in its
//! performance lanes.

#[cfg(test)]
#[path = "latency_tests.rs"]
mod latency_tests;

use crate::bit_width::{BitWidth, GroupSize};

/// Apple Silicon GPU characteristics.
#[derive(Clone, Debug, PartialEq)]
pub struct HardwareProfile {
    pub name: String,
    pub memory_bandwidth_bytes_per_second: f64,
    pub peak_float16_operations_per_second: f64,
    pub gpu_core_count: u32,
}

impl HardwareProfile {
    /// A conservative generic profile for an unrecognized Apple Silicon
    /// device, OptiQ's fallback (200 GB/s, 7 TFLOPS, 16 cores).
    #[must_use]
    pub fn conservative_fallback(device_name: &str) -> Self {
        Self {
            name: device_name.to_owned(),
            memory_bandwidth_bytes_per_second: 200.0e9,
            peak_float16_operations_per_second: 7.0e12,
            gpu_core_count: 16,
        }
    }

    /// Looks a device name up in a caller-supplied table, longest key first
    /// so `m3 max` wins over `m3`.
    #[must_use]
    pub fn from_device_name(device_name: &str, known_profiles: &[(&str, HardwareProfile)]) -> Self {
        let _ = (device_name, known_profiles);
        todo!("case-insensitive substring match, longest key first, else conservative_fallback")
    }
}

/// Per-module latency estimate.
#[derive(Clone, Debug, PartialEq)]
pub struct ModuleLatencyEstimate {
    pub module_name: String,
    pub bit_width: BitWidth,
    pub weight_bytes: f64,
    pub scale_and_bias_bytes: f64,
    pub total_bytes: f64,
    pub decode_microseconds: f64,
    pub prefill_microseconds: f64,
    pub decode_floating_point_operations: f64,
    pub arithmetic_intensity: f64,
}

/// Tunables OptiQ exposes on `estimate_layer_latency`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LatencyModelParameters {
    pub prefill_sequence_length_tokens: usize,
    /// Fixed per-module kernel dispatch cost.
    pub dispatch_overhead_microseconds: f64,
    /// Multiplier on bandwidth-limited time for dequantization compute,
    /// about 1.15 at 4 bits; 3-bit adds 8% and 2-bit 12% on top.
    pub dequantization_overhead_factor: f64,
}

impl LatencyModelParameters {
    #[must_use]
    pub const fn optiq_defaults() -> Self {
        Self {
            prefill_sequence_length_tokens: 512,
            dispatch_overhead_microseconds: 5.0,
            dequantization_overhead_factor: 1.15,
        }
    }
}

/// Bytes a quantized module reads: packed weight plus float16 scales and
/// biases per group.
#[must_use]
pub fn estimate_module_bytes(
    parameter_count: u64,
    bit_width: BitWidth,
    group_size: GroupSize,
) -> ModuleByteFootprint {
    let _ = (parameter_count, bit_width, group_size);
    todo!("weight = params * bits / 8; scale_and_bias = params / group * 4; total = sum")
}

/// The bytes a quantized module occupies, split by what they hold.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ModuleByteFootprint {
    pub weight_bytes: f64,
    pub scale_and_bias_bytes: f64,
    pub total_bytes: f64,
}

/// Floating point operations of one linear forward: `2 * tokens * in * out`.
#[must_use]
pub fn estimate_module_operations(
    input_features: u64,
    output_features: u64,
    sequence_length_tokens: usize,
) -> f64 {
    let _ = (input_features, output_features, sequence_length_tokens);
    todo!("multiply-add counts as two operations")
}

/// The full estimate for one module.
///
/// Work to port (OptiQ's `estimate_layer_latency`): decode reads the total
/// bytes plus `input_features * 2` activation bytes, divided by bandwidth,
/// times the dequantization factor (×1.08 at 3 bits, ×1.12 at 2 bits), plus
/// dispatch overhead. Prefill is operations over effective peak (×0.85 at or
/// below 3 bits, ×0.95 at 8 bits) plus dispatch overhead. Arithmetic
/// intensity is decode operations over decode bytes.
// The signature mirrors the reference estimator's inputs one for one so the
// port stays a line-by-line comparison; a parameter struct would hide that
// correspondence while the body is still being ported.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn estimate_module_latency(
    module_name: &str,
    parameter_count: u64,
    input_features: u64,
    output_features: u64,
    bit_width: BitWidth,
    group_size: GroupSize,
    hardware: &HardwareProfile,
    parameters: LatencyModelParameters,
) -> ModuleLatencyEstimate {
    let _ = (
        module_name,
        parameter_count,
        input_features,
        output_features,
        bit_width,
        group_size,
        hardware,
        parameters,
    );
    todo!("apply the roofline form documented above")
}

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

/// Infers `(input_features, output_features)` for a module from its name and
/// the model config's `hidden_size`, `num_attention_heads`,
/// `num_key_value_heads`, `head_dim`, `intermediate_size`, and `vocab_size`,
/// for Qwen3 and Llama-style naming; unknown names fall back to a square of
/// side `sqrt(parameter_count)`.
#[must_use]
pub fn infer_module_dimensions(
    module_name: &str,
    model_config: &serde_json::Value,
    parameter_count: u64,
) -> (u64, u64) {
    let _ = (module_name, model_config, parameter_count);
    todo!("match q/k/v/o, gate/up/down, lm_head, embed by suffix; else square fallback")
}
