//! Measuring a mixture-of-experts model: the router and every routed-expert
//! projection are swapped, forwarded, and restored like any other module.
//!
//! The journeys take as `model_directory` the tied synthetic checkpoint
//! `crates/quantization/tests/fixtures/build_optiq_synthetic_moe_source.py`
//! writes (a few megabytes of random weights in an original release's
//! layout, with fused expert tensors), and as `reference_artifact_path`
//! what `build_optiq_reference_artifacts.py --reference uniform_4bit
//! --methods optiq --skip-low-bit` wrote from it. A real mixture-of-experts
//! base is tens of gigabytes; the synthetic one exercises the same code in
//! seconds. Calibration text is the Romeo and Juliet fixture.

use std::path::{Path, PathBuf};

use astronomical_model_serving::quantization_probe::load_qwen3_5_probe_model;
use astronomical_quantization::artifact::converted_config::normalize_converted_config;
use astronomical_quantization::bit_width::GroupSize;
use astronomical_quantization::encoder::RangeSearchPolicy;
use astronomical_quantization::module_inventory::ModuleInventory;
use astronomical_quantization::pipeline::artifact_files::copy_source_files;
use astronomical_quantization::pipeline::{
    PipelineMethod, ReferenceMode, run_quantization_pipeline,
};
use astronomical_quantization::requantize::driver::run_requantize;
use astronomical_quantization::requantize::{
    BitPolicy, DEFAULT_COPY_PATTERNS, DEFAULT_SHARD_BYTES, RequantizeTensorPlan,
};
use astronomical_quantization::sensitivity::SensitivityProbeModel;
use astronomical_runtime_integration::MlxRuntime;
use serial_test::serial;

use super::artifact_comparison::read_json_document;
use super::exact_sweep::{probed_kl, romeo_and_juliet_samples};
use super::parity_support::{
    Qwen3_5ProbeLoader, bit_map_agreement, parity_request, rank_correlation_at,
    reference_sequence_count,
};
use super::support::{
    PREFILL_CHUNK_TOKENS, StandardErrorProgress, fresh_output_directory, journey_memory_limits,
    journey_runtime, python_reference_directory, source_model_directory,
};

/// One layer's router and its three routed-expert projections. Layer 3 is a
/// full-attention layer of the synthetic model; any layer would do.
const PROBED_MODULES: [&str; 4] = [
    "language_model.model.layers.3.mlp.gate",
    "language_model.model.layers.3.mlp.switch_mlp.gate_proj",
    "language_model.model.layers.3.mlp.switch_mlp.up_proj",
    "language_model.model.layers.3.mlp.switch_mlp.down_proj",
];

/// The same thresholds as the dense parity journey. Measured on the
/// synthetic model with two sequences: rank correlation 0.997 at 8 bits and
/// every one of 102 modules at the same width.
const MINIMUM_RANK_CORRELATION: f64 = 0.95;
const MINIMUM_BIT_MAP_AGREEMENT: f64 = 0.85;

#[test]
#[serial]
#[ignore = "needs the synthetic mixture-of-experts model_directory in the explicit test-run configuration"]
fn should_swap_and_exactly_restore_the_router_and_each_routed_expert_projection() {
    let runtime = journey_runtime();
    let output_directory = fresh_output_directory("moe-swaps");
    let bfloat16_model = bfloat16_engine_layout_copy(&runtime, &output_directory);
    let calibration_samples = romeo_and_juliet_samples(&bfloat16_model);
    let mut probe_model = load_qwen3_5_probe_model(
        &bfloat16_model,
        None,
        journey_memory_limits(),
        PREFILL_CHUNK_TOKENS,
    )
    .expect("the bf16 mixture-of-experts probe should load");
    let reference_logits = probe_model
        .forward(&calibration_samples[0])
        .expect("the reference forward should run");

    for module_name in PROBED_MODULES {
        let two_bit_kl = probed_kl(
            &mut probe_model,
            module_name,
            2,
            &calibration_samples[0],
            &reference_logits,
        );
        let restored_logits = probe_model
            .forward(&calibration_samples[0])
            .expect("the restored forward should run");
        let after_restore = probe_model
            .kl_divergence(&reference_logits, &restored_logits)
            .expect("the divergence should compute");
        eprintln!(
            "[moe-swaps] {module_name} kl@2={two_bit_kl:.3e} after_restore={after_restore:.3e}"
        );
        // The swap reaches the forward, and the restore leaves the model
        // exactly as it was, the fused gate-up owner included.
        assert!(two_bit_kl > 1e-6, "{module_name} was not swapped");
        assert!(after_restore < 1e-9, "{module_name} was not restored");
    }
    let _ = std::fs::remove_dir_all(&output_directory);
}

#[test]
#[serial]
#[ignore = "needs the synthetic mixture-of-experts model_directory and the uniform-reference reference_artifact_path the reference builder wrote from it"]
fn should_measure_a_mixture_of_experts_model_like_the_python_uniform_reference_sweep() {
    let runtime = journey_runtime();
    let output_directory = fresh_output_directory("moe-parity");
    let reference_directory = python_reference_directory().join("optiq_mixed");
    let reference_sensitivity = read_json_document(&reference_directory, "optiq/sensitivity.json");
    let mut request = parity_request(
        &output_directory,
        PipelineMethod::ExactKlSensitivity,
        reference_sequence_count(),
    );
    request.reference_mode = ReferenceMode::Uniform4Bit;

    let outcome = run_quantization_pipeline(
        &runtime,
        &request,
        u64::MAX,
        &mut Qwen3_5ProbeLoader,
        &mut StandardErrorProgress,
    )
    .expect("the measured mixture-of-experts conversion should succeed");

    let measured_sensitivity =
        read_json_document(&outcome.artifact_directory, "optiq/sensitivity.json");
    let measured_names = measured_module_names(&measured_sensitivity);
    let reference_names = measured_module_names(&reference_sensitivity);
    let rank_correlation = rank_correlation_at(&reference_sensitivity, &measured_sensitivity, 8);
    let agreement = bit_map_agreement(&reference_directory, &outcome.artifact_directory);
    eprintln!(
        "[moe-parity] modules={} rank_correlation_at_8_bits={rank_correlation:.4} same_width={:.1}%",
        measured_names.len(),
        100.0 * agreement
    );
    assert_eq!(measured_names, reference_names);
    for module_name in PROBED_MODULES {
        assert!(
            measured_names.contains(&module_name.to_owned()),
            "{module_name} was skipped"
        );
    }
    assert!(rank_correlation >= MINIMUM_RANK_CORRELATION);
    assert!(agreement >= MINIMUM_BIT_MAP_AGREEMENT);
    let _ = std::fs::remove_dir_all(&output_directory);
}

/// The module names of a sensitivity document, in document order.
fn measured_module_names(sensitivity_document: &serde_json::Value) -> Vec<String> {
    let mut module_names = Vec::new();
    for layer in sensitivity_document["layers"]
        .as_array()
        .expect("a layer list")
    {
        module_names.push(
            layer["layer_name"]
                .as_str()
                .expect("a module name")
                .to_owned(),
        );
    }
    module_names
}

/// The source at bfloat16 in the layout the engine loads, written the way
/// the pipeline writes its own bf16 probe copy.
fn bfloat16_engine_layout_copy(runtime: &MlxRuntime, output_directory: &Path) -> PathBuf {
    let source_directory = source_model_directory();
    let copy_directory = output_directory.join("bfloat16_model");
    let inventory = ModuleInventory::from_source_directory(
        &source_directory,
        GroupSize::new(64).expect("valid"),
    )
    .expect("the source should inventory");
    let plan = RequantizeTensorPlan::from_inventory(&inventory, &BitPolicy::KeepEverythingBfloat16);
    run_requantize(
        runtime,
        &source_directory,
        &copy_directory,
        &plan,
        RangeSearchPolicy::default_policy(),
        DEFAULT_SHARD_BYTES,
        &mut StandardErrorProgress,
    )
    .expect("the bf16 copy should be written");
    std::fs::copy(
        source_directory.join("config.json"),
        copy_directory.join("config.json"),
    )
    .expect("the config should copy");
    normalize_converted_config(&copy_directory.join("config.json"))
        .expect("the config should normalize");
    copy_source_files(&source_directory, &copy_directory, DEFAULT_COPY_PATTERNS)
        .expect("the tokenizer should copy");
    copy_directory
}
