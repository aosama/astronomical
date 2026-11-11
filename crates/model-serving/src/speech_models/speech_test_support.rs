//! What the speech models' MLX tests share: a small runtime, uploads,
//! read-backs, a checkpoint built from named arrays, and a tolerance check.

use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime, MlxSafetensors};
use mlx_bridge::MlxArray;

pub(crate) const ROMEO_AND_JULIET_SOURCE: &str = include_str!(
    "../../../../verification/acceptance/tests/fixtures/model_metrics_5000_romeo_and_juliet_words.txt"
);

pub(crate) fn initialize_test_runtime() -> MlxRuntime {
    MlxRuntime::initialize(
        MlxMemoryLimits::new(512 * 1024 * 1024, 512 * 1024 * 1024)
            .expect("the test memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize")
}

pub(crate) fn upload(runtime: &MlxRuntime, values: &[f32], shape: &[i32]) -> MlxArray {
    runtime
        .array_from_f32(values, shape)
        .expect("the test values upload")
}

pub(crate) fn read_back(runtime: &MlxRuntime, array: &MlxArray) -> Vec<f32> {
    runtime
        .array_to_vec_f32(array)
        .expect("the result reads back")
}

/// Values in about [-1.5, 1.5] drawn from the play's bytes, so test inputs
/// are fixed and varied without a random generator.
pub(crate) fn shakespeare_values(value_count: usize, byte_offset: usize) -> Vec<f32> {
    ROMEO_AND_JULIET_SOURCE
        .bytes()
        .skip(byte_offset)
        .take(value_count)
        .map(|source_byte| (f32::from(source_byte) - 96.0) / 32.0)
        .collect()
}

/// A safetensors map holding `named_arrays`, written to a temporary file and
/// read back the way a checkpoint is.
pub(crate) fn checkpoint_from_arrays<TensorName: AsRef<str>>(
    runtime: &MlxRuntime,
    named_arrays: &[(TensorName, MlxArray)],
) -> MlxSafetensors {
    let checkpoint_directory = tempfile::tempdir().expect("a temporary directory");
    let checkpoint_path = checkpoint_directory.path().join("checkpoint.safetensors");
    let array_refs: Vec<(&str, &MlxArray)> = named_arrays
        .iter()
        .map(|(tensor_name, array)| (tensor_name.as_ref(), array))
        .collect();
    let output_file =
        std::fs::File::create(&checkpoint_path).expect("the test checkpoint is created");
    runtime
        .save_safetensors(output_file, &array_refs, &[])
        .expect("the test checkpoint is written");
    let checkpoint_file = std::fs::File::open(&checkpoint_path).expect("the test checkpoint opens");
    runtime
        .load_safetensors(checkpoint_file, None)
        .expect("the test checkpoint loads")
}

pub(crate) fn assert_close(found: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(found.len(), expected.len(), "{found:?} vs {expected:?}");
    for (found_value, expected_value) in found.iter().zip(expected) {
        assert!(
            (found_value - expected_value).abs() <= tolerance,
            "{found:?} vs {expected:?}"
        );
    }
}
