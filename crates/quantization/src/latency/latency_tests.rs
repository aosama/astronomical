//! Unit tests for the roofline arithmetic.

use super::*;
use crate::bit_width::{BitWidth, GroupSize};

fn bits(value: u32) -> BitWidth {
    BitWidth::new(value).expect("valid")
}

#[test]
fn should_count_packed_weight_bytes_plus_float16_scales_and_biases() {
    let footprint = estimate_module_bytes(4096, bits(4), GroupSize::new(64).expect("valid"));

    assert_eq!(
        footprint,
        ModuleByteFootprint {
            weight_bytes: 2048.0,
            scale_and_bias_bytes: 256.0,
            total_bytes: 2304.0,
        }
    );
}

#[test]
fn should_predict_faster_decode_at_lower_bit_widths_on_the_same_hardware() {
    let hardware = HardwareProfile::conservative_fallback("test device");

    let four_bit = estimate_module_latency(
        "model.layers.0.mlp.up_proj",
        2048 * 8192,
        2048,
        8192,
        bits(4),
        GroupSize::new(64).expect("valid"),
        &hardware,
        LatencyModelParameters::optiq_defaults(),
    );
    let eight_bit = estimate_module_latency(
        "model.layers.0.mlp.up_proj",
        2048 * 8192,
        2048,
        8192,
        bits(8),
        GroupSize::new(64).expect("valid"),
        &hardware,
        LatencyModelParameters::optiq_defaults(),
    );

    assert!(four_bit.decode_microseconds < eight_bit.decode_microseconds);
}

#[test]
fn should_prefer_the_longest_matching_profile_key() {
    let known = [
        ("m3", HardwareProfile::conservative_fallback("M3")),
        ("m3 max", HardwareProfile::conservative_fallback("M3 Max")),
    ];

    let profile = HardwareProfile::from_device_name("Apple M3 Max", &known);

    assert_eq!(profile.name, "M3 Max");
}

#[test]
fn should_learn_a_non_negative_overhead_from_one_measurement() {
    let mut model = CalibratedLatencyModel {
        hardware: HardwareProfile::conservative_fallback("test device"),
        overhead_microseconds_per_token: 0.0,
        is_calibrated: false,
    };

    // 20 tokens per second is 50,000 microseconds per token; 30,000 of
    // linear work leaves 20,000 of overhead.
    model.calibrate(30_000.0, 20.0);

    assert!(model.is_calibrated);
    assert_eq!(model.overhead_microseconds_per_token, 20_000.0);
}

#[test]
fn should_infer_attention_and_feed_forward_dimensions_from_the_config() {
    let config = serde_json::json!({
        "hidden_size": 2048, "num_attention_heads": 16, "num_key_value_heads": 4,
        "intermediate_size": 6144, "vocab_size": 1000
    });

    assert_eq!(
        infer_module_dimensions("model.layers.0.self_attn.q_proj", &config, 0),
        (2048, 2048)
    );
    assert_eq!(
        infer_module_dimensions("model.layers.0.self_attn.k_proj", &config, 0),
        (2048, 512)
    );
    assert_eq!(
        infer_module_dimensions("model.layers.0.mlp.down_proj", &config, 0),
        (6144, 2048)
    );
    assert_eq!(
        infer_module_dimensions("model.embed_tokens", &config, 0),
        (1000, 2048)
    );
}

#[test]
fn should_fall_back_to_a_square_for_an_unrecognized_module() {
    let dimensions =
        infer_module_dimensions("model.layers.0.mixer.in_proj", &serde_json::json!({}), 4096);

    assert_eq!(dimensions, (64, 64));
}

#[test]
fn should_use_the_conservative_profile_for_an_unknown_device() {
    let known = [("m4", HardwareProfile::conservative_fallback("M4"))];

    let profile = HardwareProfile::from_device_name("Unreleased Chip", &known);

    assert_eq!(profile.name, "Unreleased Chip");
    assert_eq!(profile.memory_bandwidth_bytes_per_second, 200.0e9);
}

#[test]
fn should_clamp_the_overhead_at_zero_when_the_measurement_beats_the_linear_estimate() {
    let mut model = CalibratedLatencyModel {
        hardware: HardwareProfile::conservative_fallback("test device"),
        overhead_microseconds_per_token: 0.0,
        is_calibrated: false,
    };

    model.calibrate(80_000.0, 20.0);

    assert_eq!(model.overhead_microseconds_per_token, 0.0);
    assert_eq!(model.predict_tokens_per_second(50_000.0), 20.0);
    assert_eq!(model.predict_tokens_per_second(0.0), 0.0);
}

#[test]
fn should_run_prefill_slower_at_two_bits_than_at_four() {
    let two_bit = up_projection_estimate(bits(2));
    let four_bit = up_projection_estimate(bits(4));

    assert!(two_bit.prefill_microseconds > four_bit.prefill_microseconds);
}

fn up_projection_estimate(bit_width: BitWidth) -> ModuleLatencyEstimate {
    estimate_module_latency(
        "model.layers.0.mlp.up_proj",
        2048 * 8192,
        2048,
        8192,
        bit_width,
        GroupSize::new(64).expect("valid"),
        &HardwareProfile::conservative_fallback("test device"),
        LatencyModelParameters::optiq_defaults(),
    )
}
