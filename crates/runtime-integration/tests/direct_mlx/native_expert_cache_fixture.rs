//! Shared two-expert data used by native cache behavior tests.
//!
//! Keeping fixture construction here leaves each test file focused on one cache
//! behavior and keeps the production-shaped cache tests below the source limit.

use astronomical_runtime_integration::{
    MlxArray, MlxDtype, MlxNativeExpertParameter, MlxNativeExpertProjection,
    MlxNativeExpertTensorSourceDescriptor, MlxRuntime,
};

pub(super) const EXPERT_CAPACITY: usize = 2;
pub(super) const INPUT_DIMENSION: i32 = 64;
pub(super) const OUTPUT_DIMENSION: i32 = 64;
const QUANTIZATION_GROUP_SIZE: i32 = 64;
const QUANTIZATION_BITS: i32 = 4;
const PACKED_WORDS_PER_OUTPUT_ROW: i32 = INPUT_DIMENSION * QUANTIZATION_BITS / i32::BITS as i32;
const PACKED_WORDS_PER_EXPERT: usize = (OUTPUT_DIMENSION * PACKED_WORDS_PER_OUTPUT_ROW) as usize;
const QUANTIZATION_VALUES_PER_EXPERT: usize = OUTPUT_DIMENSION as usize;

pub(super) fn build_two_expert_source_fixture(
    source_file_path: &std::path::Path,
) -> (Vec<u8>, Vec<MlxNativeExpertTensorSourceDescriptor>) {
    let mut source_file_bytes = Vec::new();
    let mut source_descriptors = Vec::new();
    for projection in [
        MlxNativeExpertProjection::Gate,
        MlxNativeExpertProjection::Up,
        MlxNativeExpertProjection::Down,
    ] {
        let packed_weight_offset = source_file_bytes.len() as u64;
        append_u32_values(
            &mut source_file_bytes,
            &[
                vec![0x1111_1111; PACKED_WORDS_PER_EXPERT],
                vec![0x3333_3333; PACKED_WORDS_PER_EXPERT],
            ]
            .concat(),
        );
        source_descriptors.push(MlxNativeExpertTensorSourceDescriptor::new(
            projection,
            MlxNativeExpertParameter::PackedWeight,
            QUANTIZATION_GROUP_SIZE,
            QUANTIZATION_BITS,
            source_file_path.to_path_buf(),
            packed_weight_offset,
            PACKED_WORDS_PER_EXPERT * u32::BITS as usize / 8,
            vec![1, OUTPUT_DIMENSION, PACKED_WORDS_PER_OUTPUT_ROW],
            MlxDtype::UInt32,
        ));

        let scales_offset = source_file_bytes.len() as u64;
        append_f32_values(
            &mut source_file_bytes,
            &vec![1.0; QUANTIZATION_VALUES_PER_EXPERT * EXPERT_CAPACITY],
        );
        source_descriptors.push(MlxNativeExpertTensorSourceDescriptor::new(
            projection,
            MlxNativeExpertParameter::Scales,
            QUANTIZATION_GROUP_SIZE,
            QUANTIZATION_BITS,
            source_file_path.to_path_buf(),
            scales_offset,
            QUANTIZATION_VALUES_PER_EXPERT * size_of::<f32>(),
            vec![1, OUTPUT_DIMENSION, 1],
            MlxDtype::Float32,
        ));

        let biases_offset = source_file_bytes.len() as u64;
        append_f32_values(
            &mut source_file_bytes,
            &vec![0.0; QUANTIZATION_VALUES_PER_EXPERT * EXPERT_CAPACITY],
        );
        source_descriptors.push(MlxNativeExpertTensorSourceDescriptor::new(
            projection,
            MlxNativeExpertParameter::Biases,
            QUANTIZATION_GROUP_SIZE,
            QUANTIZATION_BITS,
            source_file_path.to_path_buf(),
            biases_offset,
            QUANTIZATION_VALUES_PER_EXPERT * size_of::<f32>(),
            vec![1, OUTPUT_DIMENSION, 1],
            MlxDtype::Float32,
        ));
    }
    (source_file_bytes, source_descriptors)
}

pub(super) fn build_stacked_reference(
    runtime: &MlxRuntime,
    activations: &MlxArray,
    selected_expert_indices: &MlxArray,
) -> MlxArray {
    let stacked_packed_weights = runtime
        .array_from_u32(
            &[
                vec![0x1111_1111; PACKED_WORDS_PER_EXPERT],
                vec![0x3333_3333; PACKED_WORDS_PER_EXPERT],
            ]
            .concat(),
            &[
                EXPERT_CAPACITY as i32,
                OUTPUT_DIMENSION,
                PACKED_WORDS_PER_OUTPUT_ROW,
            ],
        )
        .expect("the stacked packed weights should be valid");
    let stacked_scales = runtime
        .array_from_f32(
            &vec![1.0; QUANTIZATION_VALUES_PER_EXPERT * EXPERT_CAPACITY],
            &[EXPERT_CAPACITY as i32, OUTPUT_DIMENSION, 1],
        )
        .expect("the stacked scales should be valid");
    let stacked_biases = runtime
        .array_from_f32(
            &vec![0.0; QUANTIZATION_VALUES_PER_EXPERT * EXPERT_CAPACITY],
            &[EXPERT_CAPACITY as i32, OUTPUT_DIMENSION, 1],
        )
        .expect("the stacked biases should be valid");
    runtime
        .gather_quantized_matmul_affine(
            activations,
            &stacked_packed_weights,
            &stacked_scales,
            &stacked_biases,
            None,
            Some(selected_expert_indices),
            true,
            QUANTIZATION_GROUP_SIZE,
            QUANTIZATION_BITS,
            false,
        )
        .expect("the stacked affine reference should build")
}

fn append_u32_values(destination: &mut Vec<u8>, values: &[u32]) {
    for value in values {
        destination.extend_from_slice(&value.to_ne_bytes());
    }
}

fn append_f32_values(destination: &mut Vec<u8>, values: &[f32]) {
    for value in values {
        destination.extend_from_slice(&value.to_ne_bytes());
    }
}
