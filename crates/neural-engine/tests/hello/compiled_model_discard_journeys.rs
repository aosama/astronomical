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

use super::projection_fixture::ProjectionFixture;

// Small, because the journey is about files, not speed.
const INPUT_CHANNELS: usize = 512;
const OUTPUT_CHANNELS: usize = 256;
const TOKEN_COUNT: usize = 64;
const SINGLE_CONTRACTION_PART: usize = 1;
const WEIGHT_SEED: u64 = 0x5eed_d15c;
const MINIMUM_COSINE_SIMILARITY: f64 = 0.9999;
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
