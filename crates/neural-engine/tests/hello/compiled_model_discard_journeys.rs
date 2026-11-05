//! Does a loaded program keep predicting after its compiled copy is removed?
//!
//! Core ML writes a compiled copy of every model into the temporary directory
//! and the bridge removes it on release. A worker the daemon stops with SIGKILL
//! never releases, so serving removes the copy right after a program's first
//! run instead. This journey proves that removal costs nothing: the entry
//! leaves the temporary directory and the program's next prediction returns
//! the same values as the one before.

use std::fs;
use std::path::Path;

use astronomical_neural_engine::core_ml_projection::{CoreMlComputeUnits, CoreMlProjection};
use astronomical_neural_engine::deterministic_values::{
    PROJECTION_WEIGHT_STANDARD_DEVIATION, bell_shaped_values,
};
use astronomical_neural_engine::half_precision::{f16_bits_slice_to_f32, f32_slice_to_f16_bits};
use astronomical_neural_engine::ml_program_projection_document::MlProgramWeights;
use astronomical_neural_engine::numerical_comparison::compare_values;
use astronomical_neural_engine::projection_shape::ProjectionShape;
use serial_test::serial;

use crate::support::journey_constants::MINIMUM_COSINE_SIMILARITY;
use crate::support::projection_fixture::ProjectionFixture;

// Small, because the journey is about files, not speed.
const INPUT_CHANNELS: usize = 512;
const OUTPUT_CHANNELS: usize = 256;
const TOKEN_COUNT: usize = 64;
const SINGLE_CONTRACTION_PART: usize = 1;
const WEIGHT_SEED: u64 = 0x5eed_d15c;
/// What Core ML names the compiled copy of `projection.mlpackage`.
const COMPILED_MODEL_PREFIX: &str = "projection_";
const COMPILED_MODEL_EXTENSION: &str = "mlmodelc";

fn compiled_model_count(temporary_directory: &Path) -> usize {
    let mut compiled_model_count = 0;
    for entry in fs::read_dir(temporary_directory).expect("the temporary directory lists") {
        let entry_name = entry.expect("the entry reads").file_name();
        let entry_name = entry_name.to_string_lossy();
        if entry_name.starts_with(COMPILED_MODEL_PREFIX)
            && entry_name.ends_with(COMPILED_MODEL_EXTENSION)
        {
            compiled_model_count += 1;
        }
    }
    compiled_model_count
}

fn predicted_values(projection: &mut CoreMlProjection) -> Vec<u16> {
    projection.predict().expect("the prediction runs");
    projection.read_projected().expect("the result reads")
}

#[test]
#[serial(neural_engine)]
#[ignore = "loads one small program on the real Neural Engine and removes its compiled copy"]
fn should_keep_predicting_after_discarding_the_compiled_model() {
    let shape =
        ProjectionShape::new(INPUT_CHANNELS, OUTPUT_CHANNELS, TOKEN_COUNT).expect("valid shape");
    let fp16_weights = f32_slice_to_f16_bits(&bell_shaped_values(
        WEIGHT_SEED,
        shape.weight_element_count(),
        PROJECTION_WEIGHT_STANDARD_DEVIATION,
    ));
    let fixture = ProjectionFixture::with_fp16_weights(shape, fp16_weights);
    let temporary_directory = std::env::temp_dir();
    let package_directory = tempfile::tempdir().expect("a scratch directory for the model");
    let count_before_load = compiled_model_count(&temporary_directory);

    let mut projection = CoreMlProjection::load_ml_program(
        shape,
        MlProgramWeights::Fp16(&fixture.fp16_weights),
        CoreMlComputeUnits::CpuAndNeuralEngine,
        SINGLE_CONTRACTION_PART,
        package_directory.path(),
    )
    .expect("the program loads");
    projection
        .write_activations(&fixture.fp16_activations)
        .expect("the activations fit");
    let values_with_files = predicted_values(&mut projection);
    assert_eq!(
        compiled_model_count(&temporary_directory),
        count_before_load + 1,
        "loading writes one compiled copy into the temporary directory"
    );

    drop(package_directory);
    projection.discard_compiled_model_files();
    assert_eq!(
        compiled_model_count(&temporary_directory),
        count_before_load,
        "the compiled copy leaves the temporary directory"
    );
    let values_without_files = predicted_values(&mut projection);
    assert_eq!(
        values_without_files, values_with_files,
        "the program predicts the same values without its files"
    );

    // Discarding twice is harmless, and the program still runs after it.
    projection.discard_compiled_model_files();
    let values_after_second_discard = predicted_values(&mut projection);
    assert_eq!(values_after_second_discard, values_with_files);

    let comparison = compare_values(
        &fixture.reference_projected,
        &f16_bits_slice_to_f32(&values_without_files),
    )
    .expect("same length");
    eprintln!(
        "[neural-engine-hello] discard journey: cosine {:.8} against the fp32 reference, {} compiled copies in the temporary directory",
        comparison.cosine_similarity,
        compiled_model_count(&temporary_directory),
    );
    assert!(comparison.cosine_similarity >= MINIMUM_COSINE_SIMILARITY);
    assert_eq!(comparison.non_finite_candidate_count, 0);
}

#[test]
#[serial(neural_engine)]
#[ignore = "keeps one small program's compiled copy and loads it again on the Neural Engine"]
fn should_predict_the_same_values_from_a_kept_compiled_model() {
    let shape =
        ProjectionShape::new(INPUT_CHANNELS, OUTPUT_CHANNELS, TOKEN_COUNT).expect("valid shape");
    let fp16_weights = f32_slice_to_f16_bits(&bell_shaped_values(
        WEIGHT_SEED,
        shape.weight_element_count(),
        PROJECTION_WEIGHT_STANDARD_DEVIATION,
    ));
    let fixture = ProjectionFixture::with_fp16_weights(shape, fp16_weights);
    let package_directory = tempfile::tempdir().expect("a scratch directory for the model");
    let cache_directory = tempfile::tempdir().expect("a scratch directory for the cache");
    let kept_model_path = cache_directory.path().join("kept.mlmodelc");
    let mut projection = CoreMlProjection::load_ml_program(
        shape,
        MlProgramWeights::Fp16(&fixture.fp16_weights),
        CoreMlComputeUnits::CpuAndNeuralEngine,
        SINGLE_CONTRACTION_PART,
        package_directory.path(),
    )
    .expect("the program loads");
    projection
        .write_activations(&fixture.fp16_activations)
        .expect("the activations fit");
    let values_before = predicted_values(&mut projection);

    projection
        .keep_compiled_model_files_at(&kept_model_path)
        .expect("the compiled copy moves");
    assert!(kept_model_path.is_dir(), "the compiled copy is kept");
    assert!(
        projection
            .keep_compiled_model_files_at(&cache_directory.path().join("again.mlmodelc"))
            .is_err(),
        "there is nothing left to keep a second time"
    );
    assert_eq!(predicted_values(&mut projection), values_before);
    drop(projection);
    assert!(kept_model_path.is_dir(), "release leaves a kept copy alone");

    drop(package_directory);
    let mut reloaded = CoreMlProjection::load_compiled_model(
        &kept_model_path,
        shape,
        CoreMlComputeUnits::CpuAndNeuralEngine,
    )
    .expect("the kept copy loads");
    reloaded
        .write_activations(&fixture.fp16_activations)
        .expect("the activations fit");
    assert_eq!(predicted_values(&mut reloaded), values_before);
    drop(reloaded);
    assert!(
        kept_model_path.is_dir(),
        "a loaded copy stays with its owner"
    );
}

#[test]
#[serial(neural_engine)]
#[ignore = "compiles one small program straight to a kept path and loads it from there"]
fn should_predict_the_same_values_from_a_program_compiled_to_its_final_path() {
    let shape =
        ProjectionShape::new(INPUT_CHANNELS, OUTPUT_CHANNELS, TOKEN_COUNT).expect("valid shape");
    let fixture = ProjectionFixture::with_shape(INPUT_CHANNELS, OUTPUT_CHANNELS, TOKEN_COUNT);
    let package_directory = tempfile::tempdir().expect("a scratch directory for the package");
    let kept_directory = tempfile::tempdir().expect("a directory for the kept program");
    // A parent that does not exist yet, as a cache folder is on its first use.
    let kept_model_path = kept_directory.path().join("nested").join("kept.mlmodelc");

    CoreMlProjection::compile_ml_program_to(
        shape,
        MlProgramWeights::Fp16(&fixture.fp16_weights),
        SINGLE_CONTRACTION_PART,
        package_directory.path(),
        &kept_model_path,
    )
    .expect("the program compiles to its final path");
    let mut kept_program = CoreMlProjection::load_compiled_model(
        &kept_model_path,
        shape,
        CoreMlComputeUnits::CpuAndNeuralEngine,
    )
    .expect("the kept program loads");
    kept_program
        .write_activations(&fixture.fp16_activations)
        .expect("the activations fit");
    let kept_values = predicted_values(&mut kept_program);

    let comparison = compare_values(
        &fixture.reference_projected,
        &f16_bits_slice_to_f32(&kept_values),
    )
    .expect("same length");
    assert!(comparison.cosine_similarity >= MINIMUM_COSINE_SIMILARITY);
    assert!(kept_model_path.is_dir());
}

#[test]
#[serial(neural_engine)]
#[ignore = "asks Core ML to load a kept program that was never compiled"]
fn should_refuse_to_load_a_kept_program_that_was_never_compiled() {
    let shape =
        ProjectionShape::new(INPUT_CHANNELS, OUTPUT_CHANNELS, TOKEN_COUNT).expect("valid shape");
    let kept_directory = tempfile::tempdir().expect("a directory for the kept program");
    let kept_model_path = kept_directory.path().join("kept.mlmodelc");

    let loaded = CoreMlProjection::load_compiled_model(
        &kept_model_path,
        shape,
        CoreMlComputeUnits::CpuAndNeuralEngine,
    );

    assert!(loaded.is_err());
    assert!(!kept_model_path.exists());
}
