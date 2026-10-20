//! Unit tests for the roofline arithmetic.

use super::*;

fn bits(value: u32) -> BitWidth {
    BitWidth::new(value).expect("valid")
}

#[test]
#[ignore = "pending implementation of estimate_module_bytes"]
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
#[ignore = "pending implementation of estimate_module_latency"]
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
#[ignore = "pending implementation of HardwareProfile::from_device_name"]
fn should_prefer_the_longest_matching_profile_key() {
    let known = [
        ("m3", HardwareProfile::conservative_fallback("M3")),
        ("m3 max", HardwareProfile::conservative_fallback("M3 Max")),
    ];

    let profile = HardwareProfile::from_device_name("Apple M3 Max", &known);

    assert_eq!(profile.name, "M3 Max");
}

#[test]
#[ignore = "pending implementation of CalibratedLatencyModel::calibrate"]
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
