//! Builds small synthetic weight files for tests that need a graphics processor
//! but no installed model.
//!
//! Values are drawn from the Romeo and Juliet fixture, quantized where a real
//! artifact would be, written to a temporary safetensors file and loaded
//! exactly as a real artifact is loaded.

use std::fs::File;

use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime, MlxSafetensors};
use mlx_c_rust::MlxArray;

const ROMEO_AND_JULIET_SOURCE: &str = include_str!(
    "../../../verification/acceptance/tests/fixtures/model_metrics_5000_romeo_and_juliet_words.txt"
);
pub(crate) const SYNTHETIC_QUANTIZATION_GROUP_SIZE: i32 = 32;
pub(crate) const SYNTHETIC_QUANTIZATION_BITS: i32 = 8;

pub(crate) fn initialize_test_runtime() -> MlxRuntime {
    MlxRuntime::initialize(
        MlxMemoryLimits::new(512 * 1024 * 1024, 512 * 1024 * 1024)
            .expect("the test memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize")
}

/// Float32 values within about 0.05 of `center_value`, drawn from the play's bytes.
///
/// The spread is kept small on purpose. With weights several times larger
/// a few layers amplify a last-bit difference between two kernel shapes
/// into the third decimal, which would hide a real masking error behind a
/// loose tolerance.
pub(crate) fn shakespeare_values(
    runtime: &MlxRuntime,
    shape: &[i32],
    byte_offset: usize,
    center_value: f32,
) -> MlxArray {
    let value_count = shape.iter().product::<i32>() as usize;
    let source_bytes = ROMEO_AND_JULIET_SOURCE.as_bytes();
    let mut values = Vec::with_capacity(value_count);
    for value_index in 0..value_count {
        let source_byte = source_bytes[(byte_offset + value_index) % source_bytes.len()];
        values.push(center_value + (f32::from(source_byte) - 96.0) / 640.0);
    }
    runtime
        .array_from_f32(&values, shape)
        .expect("the fixture values should upload")
}

/// The named tensors of a synthetic model, filled one layer at a time.
pub(crate) struct SyntheticWeights {
    pub(crate) named_tensors: Vec<(String, MlxArray)>,
    next_byte_offset: usize,
}

impl SyntheticWeights {
    pub(crate) fn new() -> Self {
        Self {
            named_tensors: Vec::new(),
            next_byte_offset: 0,
        }
    }

    /// Adds a dense tensor whose values sit near `center_value`.
    pub(crate) fn push_dense(
        &mut self,
        runtime: &MlxRuntime,
        tensor_name: &str,
        shape: &[i32],
        center_value: f32,
    ) {
        let dense_tensor = shakespeare_values(runtime, shape, self.next_byte_offset, center_value);
        self.next_byte_offset += shape.iter().product::<i32>() as usize;
        self.named_tensors
            .push((tensor_name.to_owned(), dense_tensor));
    }

    /// Adds an affine-quantized `[outputs, inputs]` weight as its `weight`,
    /// `scales` and `biases` tensors under `tensor_prefix`.
    pub(crate) fn push_quantized(
        &mut self,
        runtime: &MlxRuntime,
        tensor_prefix: &str,
        shape: [i32; 2],
    ) {
        let dense_weight = shakespeare_values(runtime, &shape, self.next_byte_offset, 0.0);
        self.next_byte_offset += (shape[0] * shape[1]) as usize;
        let (packed_weight, scales, biases) = runtime
            .quantize_affine(
                &dense_weight,
                SYNTHETIC_QUANTIZATION_GROUP_SIZE,
                SYNTHETIC_QUANTIZATION_BITS,
            )
            .expect("the fixture weight should quantize");
        self.named_tensors
            .push((format!("{tensor_prefix}.weight"), packed_weight));
        self.named_tensors
            .push((format!("{tensor_prefix}.scales"), scales));
        self.named_tensors
            .push((format!("{tensor_prefix}.biases"), biases));
    }
}

/// Writes the tensors to a temporary file and loads them as a real artifact is loaded.
pub(crate) fn load_synthetic_weights(
    runtime: &MlxRuntime,
    named_tensors: &[(String, MlxArray)],
) -> MlxSafetensors {
    let mut borrowed_tensors = Vec::with_capacity(named_tensors.len());
    for (tensor_name, tensor) in named_tensors {
        borrowed_tensors.push((tensor_name.as_str(), tensor));
    }
    let weights_file = tempfile::tempfile().expect("a temporary weights file");
    let reopened_weights_file: File = weights_file
        .try_clone()
        .expect("the temporary file should reopen");
    runtime
        .save_safetensors(weights_file, &borrowed_tensors, &[])
        .expect("the synthetic weights should save");
    runtime
        .load_safetensors(reopened_weights_file, None)
        .expect("the synthetic weights should load")
}

pub(crate) fn largest_difference(left_values: &[f32], right_values: &[f32]) -> f32 {
    assert_eq!(left_values.len(), right_values.len());
    let mut largest_difference = 0.0_f32;
    for (left_value, right_value) in left_values.iter().zip(right_values) {
        largest_difference = largest_difference.max((left_value - right_value).abs());
    }
    largest_difference
}
