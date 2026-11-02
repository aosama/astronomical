//! `static` method end to end, then served by Astronomical's own engine.

use std::path::Path;

use astronomical_quantization::bit_width::GroupSize;
use astronomical_quantization::error::QuantizationError;
use astronomical_quantization::pipeline::{
    PipelineMethod, ProbeModelHandle, ProbeModelLoader, QuantizationPipelineRequest, ReferenceMode,
    run_quantization_pipeline,
};
use serial_test::serial;

use super::served_generation::generate_greedy_continuation;
use super::support::{
    StandardErrorProgress, fresh_output_directory, journey_runtime, source_model_directory,
};

#[test]
#[serial]
#[ignore = "needs model_directory (a small Qwen3.5 checkpoint) in the explicit test-run configuration"]
fn should_convert_with_structural_rules_and_generate_a_romeo_and_juliet_continuation() {
    let runtime = journey_runtime();
    let output_directory = fresh_output_directory("static");
    let request = static_request(&source_model_directory(), &output_directory);
    let mut progress_reporter = StandardErrorProgress;

    let outcome = run_quantization_pipeline(
        &runtime,
        &request,
        u64::MAX,
        &mut StaticPathNeverLoadsAProbe,
        &mut progress_reporter,
    )
    .expect("the static conversion should succeed");
    eprintln!(
        "[static-conversion] artifact={} achieved_bits_per_weight={:.2} shards={} bytes={}",
        outcome.artifact_directory.display(),
        outcome.allocation.achieved_bits_per_weight,
        outcome.shard_count,
        outcome.total_output_bytes
    );
    drop(runtime);

    let continuation = generate_greedy_continuation(&outcome.artifact_directory);

    eprintln!("[static-conversion] continuation={continuation:?}");
    assert!(!continuation.trim().is_empty());
    assert!(!continuation.contains('\u{FFFD}'));
    let _ = std::fs::remove_dir_all(&output_directory);
}

fn static_request(source: &Path, output: &Path) -> QuantizationPipelineRequest {
    QuantizationPipelineRequest {
        source_directory: source.to_path_buf(),
        output_directory: output.to_path_buf(),
        base_model_name: "acceptance/static".to_owned(),
        method: PipelineMethod::StaticStructural,
        reference_mode: ReferenceMode::Auto,
        target_bits_per_weight: 5.0,
        candidate_bits: vec![4, 8],
        group_size: GroupSize::new(64).expect("valid"),
        calibration_sequence_count: 0,
        calibration_sequence_length_tokens: 512,
        calibration_mix_path: None,
        floor_components_per_block: 2,
        range_search_policy_enabled: true,
        range_search_maximum_bits: 3,
        overwrite_output: false,
    }
}

struct StaticPathNeverLoadsAProbe;

impl ProbeModelLoader for StaticPathNeverLoadsAProbe {
    fn load_probe_model(
        &mut self,
        _probe_directory: &Path,
        _streaming_source_directory: Option<&Path>,
    ) -> Result<Box<dyn ProbeModelHandle>, QuantizationError> {
        panic!("the static path never loads a probe model")
    }
}
