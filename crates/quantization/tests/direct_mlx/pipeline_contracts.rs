//! The pipeline end to end on a synthetic source: the static path writes a
//! complete artifact, an existing artifact is refused, and the exact path
//! drives a fake probe handle through both reference modes.

use std::path::Path;

use astronomical_quantization::bit_width::GroupSize;
use astronomical_quantization::error::QuantizationError;
use astronomical_quantization::pipeline::{
    PipelineMethod, ProbeModelHandle, ProbeModelLoader, QuantizationPipelineRequest, ReferenceMode,
    run_quantization_pipeline,
};
use astronomical_quantization::progress::SilentProgressReporter;
use astronomical_quantization::sensitivity::exact_sweep::ExactSweepRequest;
use astronomical_quantization::sensitivity::{
    SensitivityMethod, SensitivityResult, SensitivityTable,
};
use astronomical_runtime_integration::MlxRuntime;
use mlx_bridge::{MlxArray, MlxDtype};
use serial_test::parallel;

use super::support::{contract_runtime, outlier_heavy_weights};

const HIDDEN_COLUMNS: usize = 256;
const INTERMEDIATE_ROWS: usize = 128;
const UP_PROJECTION: &str = "model.layers.0.mlp.up_proj";
const DOWN_PROJECTION: &str = "model.layers.0.mlp.down_proj";

/// Every word maps to the unknown token, which is all calibration needs to
/// produce fixed-length sequences.
const WORD_LEVEL_TOKENIZER_JSON: &str = r#"{"version": "1.0", "truncation": null, "padding": null, "added_tokens": [], "normalizer": null, "pre_tokenizer": {"type": "Whitespace"}, "post_processor": null, "decoder": null, "model": {"type": "WordLevel", "vocab": {"[UNK]": 0, "the": 1}, "unk_token": "[UNK]"}}"#;

#[test]
#[parallel]
fn should_write_a_complete_static_artifact() {
    let runtime = contract_runtime();
    let workspace = tempfile::tempdir().expect("a temporary directory should be creatable");
    let request = pipeline_request(workspace.path(), PipelineMethod::StaticStructural);
    write_source(&runtime, &request.source_directory);
    let mut silent_reporter = SilentProgressReporter;

    let outcome = run_quantization_pipeline(
        &runtime,
        &request,
        u64::MAX,
        &mut UnusedProbeLoader,
        &mut silent_reporter,
    )
    .expect("the static conversion should succeed");

    let artifact = &outcome.artifact_directory;
    assert_eq!(artifact, &request.output_directory.join("static_mixed"));
    assert_eq!(outcome.method, SensitivityMethod::StaticStructuralRules);
    assert!(outcome.sensitivity_document_written);
    for expected_file in [
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "optiq/metadata.json",
        "optiq/sensitivity.json",
    ] {
        assert!(artifact.join(expected_file).is_file(), "{expected_file}");
    }
    let config = read_json(&artifact.join("config.json"));
    let metadata = read_json(&artifact.join("optiq/metadata.json"));
    for module_name in [UP_PROJECTION, DOWN_PROJECTION] {
        assert_eq!(
            config["quantization"][module_name]["bits"],
            metadata["per_layer"][module_name]["bits"]
        );
    }
}

#[test]
#[parallel]
fn should_refuse_an_existing_artifact_unless_overwrite_is_requested() {
    let runtime = contract_runtime();
    let workspace = tempfile::tempdir().expect("a temporary directory should be creatable");
    let mut request = pipeline_request(workspace.path(), PipelineMethod::StaticStructural);
    write_source(&runtime, &request.source_directory);
    std::fs::create_dir_all(request.output_directory.join("static_mixed")).expect("creatable");
    let mut silent_reporter = SilentProgressReporter;

    let refusal = run_quantization_pipeline(
        &runtime,
        &request,
        u64::MAX,
        &mut UnusedProbeLoader,
        &mut silent_reporter,
    );
    request.overwrite_output = true;
    let overwritten = run_quantization_pipeline(
        &runtime,
        &request,
        u64::MAX,
        &mut UnusedProbeLoader,
        &mut silent_reporter,
    );

    assert!(matches!(
        refusal,
        Err(QuantizationError::OutputDirectoryExists { .. })
    ));
    assert!(overwritten.is_ok());
}

#[test]
#[parallel]
fn should_sweep_against_the_bfloat16_source_when_it_fits() {
    let runtime = contract_runtime();
    let workspace = tempfile::tempdir().expect("a temporary directory should be creatable");
    let request = pipeline_request(workspace.path(), PipelineMethod::ExactKlSensitivity);
    write_source(&runtime, &request.source_directory);
    let mut silent_reporter = SilentProgressReporter;

    let outcome = run_quantization_pipeline(
        &runtime,
        &request,
        u64::MAX,
        &mut Bfloat16ReferenceLoader,
        &mut silent_reporter,
    )
    .expect("the exact conversion should succeed");

    assert_eq!(outcome.method, SensitivityMethod::ExactBfloat16Reference);
    assert_eq!(
        outcome.artifact_directory,
        request.output_directory.join("optiq_mixed")
    );
}

#[test]
#[parallel]
fn should_sweep_against_a_uniform_baseline_and_remove_it_when_bfloat16_does_not_fit() {
    let runtime = contract_runtime();
    let workspace = tempfile::tempdir().expect("a temporary directory should be creatable");
    let request = pipeline_request(workspace.path(), PipelineMethod::ExactKlSensitivity);
    write_source(&runtime, &request.source_directory);
    let mut silent_reporter = SilentProgressReporter;

    let outcome = run_quantization_pipeline(
        &runtime,
        &request,
        1,
        &mut UniformReferenceLoader,
        &mut silent_reporter,
    )
    .expect("the exact conversion should succeed");

    assert_eq!(outcome.method, SensitivityMethod::ExactUniform4BitReference);
    assert!(
        !request
            .output_directory
            .join("_uniform_4bit_baseline")
            .exists()
    );
}

/// A probe that returns a fixed table without forwarding anything, so the
/// contract exercises the pipeline's sequencing rather than a model.
struct FixedTableProbe;

impl ProbeModelHandle for FixedTableProbe {
    fn run_sweep(
        &mut self,
        request: ExactSweepRequest<'_>,
    ) -> Result<SensitivityTable, QuantizationError> {
        assert!(!request.calibration_samples.is_empty());
        assert!(request.checkpoint_path.is_some());
        let mut results = Vec::new();
        for module_name in [UP_PROJECTION, DOWN_PROJECTION] {
            let mut kl_divergence_by_bit_width = std::collections::BTreeMap::new();
            for bit_width in request.candidate_bit_widths.as_slice() {
                kl_divergence_by_bit_width.insert(*bit_width, 1.0 / f64::from(bit_width.bits()));
            }
            results.push(SensitivityResult {
                module_name: module_name.to_owned(),
                kl_divergence_by_bit_width,
                logical_parameter_count: (INTERMEDIATE_ROWS * HIDDEN_COLUMNS) as u64,
            });
        }
        Ok(SensitivityTable {
            method: request.method,
            results,
        })
    }
}

struct UnusedProbeLoader;

impl ProbeModelLoader for UnusedProbeLoader {
    fn load_probe_model(
        &mut self,
        _probe_directory: &Path,
        _streaming_source_directory: Option<&Path>,
    ) -> Result<Box<dyn ProbeModelHandle>, QuantizationError> {
        panic!("the static path never loads a probe model")
    }
}

struct Bfloat16ReferenceLoader;

impl ProbeModelLoader for Bfloat16ReferenceLoader {
    fn load_probe_model(
        &mut self,
        probe_directory: &Path,
        streaming_source_directory: Option<&Path>,
    ) -> Result<Box<dyn ProbeModelHandle>, QuantizationError> {
        assert!(probe_directory.ends_with("source"));
        assert!(streaming_source_directory.is_none());
        Ok(Box::new(FixedTableProbe))
    }
}

struct UniformReferenceLoader;

impl ProbeModelLoader for UniformReferenceLoader {
    fn load_probe_model(
        &mut self,
        probe_directory: &Path,
        streaming_source_directory: Option<&Path>,
    ) -> Result<Box<dyn ProbeModelHandle>, QuantizationError> {
        assert!(probe_directory.ends_with("_uniform_4bit_baseline"));
        assert!(probe_directory.join("config.json").is_file());
        assert!(probe_directory.join("tokenizer.json").is_file());
        assert!(streaming_source_directory.is_some_and(is_source_directory));
        Ok(Box::new(FixedTableProbe))
    }
}

fn is_source_directory(directory: &Path) -> bool {
    directory.ends_with("source")
}

fn pipeline_request(workspace: &Path, method: PipelineMethod) -> QuantizationPipelineRequest {
    QuantizationPipelineRequest {
        source_directory: workspace.join("source"),
        output_directory: workspace.join("output"),
        base_model_name: "fixture/synthetic".to_owned(),
        method,
        reference_mode: ReferenceMode::Auto,
        target_bits_per_weight: 5.0,
        candidate_bits: vec![4, 8],
        group_size: GroupSize::new(64).expect("valid"),
        calibration_sequence_count: 2,
        calibration_sequence_length_tokens: 32,
        calibration_mix_path: None,
        floor_components_per_block: 2,
        range_search_policy_enabled: true,
        range_search_maximum_bits: 3,
        overwrite_output: false,
    }
}

/// Two linears and a norm in bf16, a config, and a tokenizer.
fn write_source(runtime: &MlxRuntime, source: &Path) {
    std::fs::create_dir_all(source).expect("creatable");
    let up_projection = bfloat16_weight(runtime, INTERMEDIATE_ROWS, HIDDEN_COLUMNS);
    let down_projection = bfloat16_weight(runtime, HIDDEN_COLUMNS, INTERMEDIATE_ROWS);
    let norm = runtime
        .astype(
            &runtime
                .array_from_f32(&[1.0; HIDDEN_COLUMNS], &[HIDDEN_COLUMNS as i32])
                .expect("array"),
            MlxDtype::BFloat16,
        )
        .expect("cast");
    let shard_file =
        std::fs::File::create(source.join("model.safetensors")).expect("the shard is creatable");
    runtime
        .save_safetensors(
            shard_file,
            &[
                ("model.layers.0.mlp.up_proj.weight", &up_projection),
                ("model.layers.0.mlp.down_proj.weight", &down_projection),
                ("model.layers.0.input_layernorm.weight", &norm),
            ],
            &[("format", "mlx")],
        )
        .expect("the shard should save");
    std::fs::write(source.join("config.json"), r#"{"model_type": "qwen3"}"#).expect("writable");
    std::fs::write(source.join("tokenizer.json"), WORD_LEVEL_TOKENIZER_JSON).expect("writable");
}

fn bfloat16_weight(runtime: &MlxRuntime, rows: usize, columns: usize) -> MlxArray {
    let float32_weight = runtime
        .array_from_f32(
            &outlier_heavy_weights(rows, columns),
            &[rows as i32, columns as i32],
        )
        .expect("array");
    runtime
        .astype(&float32_weight, MlxDtype::BFloat16)
        .expect("cast")
}

fn read_json(path: &Path) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(path).expect("readable")).expect("parses")
}
