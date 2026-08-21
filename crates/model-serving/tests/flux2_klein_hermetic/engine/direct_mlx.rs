//! Small-geometry qualification of MLX load boundaries, keyed noise, and Euler arithmetic.
//! Direct tests share one limit contract because MLX runtime initialization is process-wide.

use std::fs;

use astronomical_model_serving::{
    Flux2KleinArtifactProvenance, Flux2KleinImageEngine, ImageGenerationEngine,
    flux2_klein_allocator_cache_limit_for_tests, flux2_klein_euler_update_for_tests,
    flux2_klein_keyed_noise_and_euler_for_tests,
};
use astronomical_runtime_integration::{MlxDtype, MlxMemoryLimits, MlxRuntime};

use crate::common::{
    DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES, DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
};

const ROMEO_AND_JULIET_SOURCE: &str = include_str!(
    "../../../../../apps/inference-worker/tests/fixtures/model_metrics_5000_romeo_and_juliet_words.txt"
);

#[test]
fn should_restore_the_original_image_allocator_cache_policy_after_a_ceiling_increase() {
    let original_allocator_cache_limit_bytes = 384_000_000;

    let reduced_limit = flux2_klein_allocator_cache_limit_for_tests(
        original_allocator_cache_limit_bytes,
        256_000_000,
    );
    let restored_limit = flux2_klein_allocator_cache_limit_for_tests(
        original_allocator_cache_limit_bytes,
        512_000_000,
    );

    assert_eq!(reduced_limit, 256_000_000);
    assert_eq!(restored_limit, original_allocator_cache_limit_bytes);
}

#[test]
fn should_construct_the_real_artifact_factory_seam_without_loading_a_model() {
    let model_directory = tempfile::tempdir().expect("the constructor fixture should exist");
    let engine = Flux2KleinImageEngine::from_model_family_factory(
        model_directory.path(),
        Flux2KleinArtifactProvenance::official(),
        DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES,
        DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
        false,
        model_directory.path().join("performance.jsonl"),
    );

    assert_eq!(engine.loaded_revision(), None);
}

#[test]
fn should_clear_replacement_allocator_cache_and_persist_a_bounded_failed_load_report() {
    let model_directory = tempfile::tempdir().expect("the invalid artifact directory should exist");
    let attribution_log_path = model_directory.path().join("performance.jsonl");
    let mut engine = Flux2KleinImageEngine::from_model_family_factory(
        model_directory.path(),
        Flux2KleinArtifactProvenance::official(),
        DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES,
        DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
        true,
        attribution_log_path.clone(),
    );

    engine
        .load()
        .expect_err("the empty artifact should fail after replacement cache cleanup");

    let report_line = fs::read_to_string(attribution_log_path)
        .expect("failed loading should persist attribution");
    let report: serde_json::Value = serde_json::from_str(report_line.trim())
        .expect("failed loading attribution should be valid JSON");
    assert_eq!(report["report_kind"], "model_loading");
    assert_eq!(report["outcome"], "failed");
    assert!(
        report["failure_description"]
            .as_str()
            .is_some_and(|description| description.chars().count() <= 512)
    );
    assert!(
        report["operations"]
            .as_array()
            .expect("load operations should be an array")
            .iter()
            .any(|operation| operation["operation"] == "mlx_allocator_cache_cleanup"),
        "failed load report should attribute replacement cleanup: {report}"
    );
}

#[test]
fn should_preserve_bf16_and_determinism_for_small_keyed_noise_and_euler_geometry() {
    let runtime = MlxRuntime::initialize(
        MlxMemoryLimits::new(
            DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES,
            DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
        )
        .expect("the direct-MLX limits should be valid"),
    )
    .expect("the pinned MLX runtime should initialize");
    let seed = ROMEO_AND_JULIET_SOURCE
        .bytes()
        .take(8)
        .fold(0_u64, |seed, source_byte| {
            seed.rotate_left(5) ^ u64::from(source_byte)
        });

    let first = flux2_klein_keyed_noise_and_euler_for_tests(&runtime, seed, &[1, 2, 128], -0.25)
        .expect("the first reduced BF16 denoising state should evaluate");
    let second = flux2_klein_keyed_noise_and_euler_for_tests(&runtime, seed, &[1, 2, 128], -0.25)
        .expect("the repeated reduced BF16 denoising state should evaluate");
    let baseline = flux2_klein_keyed_noise_and_euler_for_tests(&runtime, seed, &[1, 2, 128], 0.0)
        .expect("the reduced keyed-noise baseline should evaluate");

    assert_eq!(first.dtype(), MlxDtype::BFloat16);
    assert_eq!(first.shape(), [1, 2, 128]);
    let first_f32 = runtime
        .astype(&first, MlxDtype::Float32)
        .expect("the first BF16 state should cast");
    let second_f32 = runtime
        .astype(&second, MlxDtype::Float32)
        .expect("the second BF16 state should cast");
    let first_values = first_f32
        .to_vec_f32()
        .expect("the first state should materialize");
    assert_eq!(
        first_values,
        second_f32
            .to_vec_f32()
            .expect("the second state should materialize")
    );
    let baseline_values = runtime
        .astype(&baseline, MlxDtype::Float32)
        .expect("the baseline should cast")
        .to_vec_f32()
        .expect("the baseline should materialize");
    for (updated_value, baseline_value) in first_values.iter().zip(baseline_values) {
        assert!((*updated_value - baseline_value + 0.25).abs() <= 0.02);
    }
}

#[test]
fn should_accumulate_the_euler_update_in_float32_before_casting_to_bf16() {
    let runtime = MlxRuntime::initialize(
        MlxMemoryLimits::new(
            DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES,
            DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
        )
        .expect("the direct-MLX limits should be valid"),
    )
    .expect("the pinned MLX runtime should initialize");
    let sample = runtime
        .full(&[1], 1.0, MlxDtype::BFloat16)
        .expect("the scalar sample should exist");
    let model_output = runtime
        .full(&[1], 1.0, MlxDtype::BFloat16)
        .expect("the scalar model output should exist");

    let updated = flux2_klein_euler_update_for_tests(&runtime, &sample, &model_output, 0.003_91)
        .expect("the scalar Euler update should evaluate");

    assert_eq!(updated.dtype(), MlxDtype::BFloat16);
    assert_eq!(
        runtime
            .astype(&updated, MlxDtype::Float32)
            .expect("the updated scalar should cast")
            .to_vec_f32()
            .expect("the updated scalar should materialize"),
        [1.007_812_5]
    );
}
