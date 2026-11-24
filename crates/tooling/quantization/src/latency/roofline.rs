//! The per-module roofline: bytes moved and operations performed at a bit
//! width, and the latency the slower of the two implies.
//!
//! Ported from OptiQ's `estimate_layer_bytes`, `estimate_layer_flops`, `estimate_layer_latency`, and `get_layer_dimensions` (`optiq/core/latency.py`).

use crate::bit_width::{BitWidth, GroupSize};

use super::hardware_profile::HardwareProfile;

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
    let weight_bytes = parameter_count as f64 * f64::from(bit_width.bits()) / 8.0;
    let group_count = parameter_count as f64 / f64::from(group_size.elements());
    let scale_and_bias_bytes = group_count * FLOAT16_BYTES * 2.0;
    ModuleByteFootprint {
        weight_bytes,
        scale_and_bias_bytes,
        total_bytes: weight_bytes + scale_and_bias_bytes,
    }
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
    2.0 * sequence_length_tokens as f64 * input_features as f64 * output_features as f64
}

/// The full estimate for one module, OptiQ's `estimate_layer_latency`:
/// decode reads the total
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
    let footprint = estimate_module_bytes(parameter_count, bit_width, group_size);
    let decode_read_bytes = footprint.total_bytes + input_features as f64 * FLOAT16_BYTES;
    let dequantization_factor =
        parameters.dequantization_overhead_factor * unpacking_slowdown(bit_width);
    let decode_seconds =
        decode_read_bytes / hardware.memory_bandwidth_bytes_per_second * dequantization_factor;
    let prefill_operations = estimate_module_operations(
        input_features,
        output_features,
        parameters.prefill_sequence_length_tokens,
    );
    let effective_operations_per_second =
        hardware.peak_float16_operations_per_second * kernel_efficiency(bit_width);
    let prefill_seconds = prefill_operations / effective_operations_per_second;
    let decode_floating_point_operations =
        estimate_module_operations(input_features, output_features, 1);
    let arithmetic_intensity = if decode_read_bytes > 0.0 {
        decode_floating_point_operations / decode_read_bytes
    } else {
        0.0
    };
    ModuleLatencyEstimate {
        module_name: module_name.to_owned(),
        bit_width,
        weight_bytes: footprint.weight_bytes,
        scale_and_bias_bytes: footprint.scale_and_bias_bytes,
        total_bytes: footprint.total_bytes,
        decode_microseconds: decode_seconds * 1e6 + parameters.dispatch_overhead_microseconds,
        prefill_microseconds: prefill_seconds * 1e6 + parameters.dispatch_overhead_microseconds,
        decode_floating_point_operations,
        arithmetic_intensity,
    }
}

/// Scales and biases, and the decode input activation, are float16.
const FLOAT16_BYTES: f64 = 2.0;

/// Extra decode cost of unpacking codes that straddle word boundaries:
/// OptiQ measured 3-bit about 8% and 2-bit about 12% slower than 4-bit.
fn unpacking_slowdown(bit_width: BitWidth) -> f64 {
    match bit_width.bits() {
        3 => 1.08,
        2 => 1.12,
        _ => 1.0,
    }
}

/// The fraction of peak a quantized matmul reaches at prefill: 3-bit and
/// below run less optimized kernels, 8-bit dequantizes a wider word.
fn kernel_efficiency(bit_width: BitWidth) -> f64 {
    match bit_width.bits() {
        bits if bits <= 3 => 0.85,
        8 => 0.95,
        _ => 1.0,
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
    let hidden_size = config_integer(model_config, "hidden_size", 1024);
    let attention_head_count = config_integer(model_config, "num_attention_heads", 16);
    let key_value_head_count = config_integer(model_config, "num_key_value_heads", 8);
    let head_dimension = config_integer(
        model_config,
        "head_dim",
        hidden_size / attention_head_count.max(1),
    );
    let intermediate_size = config_integer(model_config, "intermediate_size", 3072);
    let vocabulary_size = config_integer(model_config, "vocab_size", 151_936);
    let lowercase_name = module_name.to_lowercase();
    if lowercase_name.contains("q_proj") {
        (hidden_size, attention_head_count * head_dimension)
    } else if lowercase_name.contains("k_proj") || lowercase_name.contains("v_proj") {
        (hidden_size, key_value_head_count * head_dimension)
    } else if lowercase_name.contains("o_proj") {
        (attention_head_count * head_dimension, hidden_size)
    } else if lowercase_name.contains("gate_proj") || lowercase_name.contains("up_proj") {
        (hidden_size, intermediate_size)
    } else if lowercase_name.contains("down_proj") {
        (intermediate_size, hidden_size)
    } else if lowercase_name.contains("lm_head") {
        (hidden_size, vocabulary_size)
    } else if lowercase_name.contains("embed") {
        (vocabulary_size, hidden_size)
    } else {
        // OptiQ's square fallback; the side is the parameter count's root.
        let side = (parameter_count as f64).sqrt() as u64;
        (side, side)
    }
}

/// An unsigned config field, or OptiQ's default when absent.
fn config_integer(model_config: &serde_json::Value, field_name: &str, default_value: u64) -> u64 {
    model_config[field_name].as_u64().unwrap_or(default_value)
}
