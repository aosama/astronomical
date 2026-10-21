//! MLX operations for decoding one bounded jangtq2 expert page.
//!
//! The CPU format contract owns bit extraction and cubic codebook semantics;
//! this module mirrors those rules with MLX arrays so selected expert pages
//! can be decoded on the graphics processor without a full weight-stack copy
//! through host memory.

use astronomical_mlx_c_rust::{MlxArray, MlxDtype};
use astronomical_runtime_integration::{MlxRuntime, MlxRuntimeError};

use super::codebook;
use super::hadamard32;

const CODEBOOK_DECODE_OPERATION: &str = "decode jangtq2 codebook expert rows";
const HADAMARD_ACTIVATION_ROTATION_OPERATION: &str = "rotate jangtq2 expert activations";

/// Maximum number of full-width temporary arrays held simultaneously.
///
/// The high-word shift overlaps the retained low-word result with its source
/// and output; subsequent bitwise stages also hold two inputs and one output.
/// The final bfloat16 page is accounted for as persisted output, not workspace.
/// The six one-dimensional index arrays are accounted for separately by page
/// admission, including when a synthetic page has only one row.
pub const MAXIMUM_SIMULTANEOUS_DECODE_TEMPORARY_ARRAY_COUNT: usize = 3;

/// Maximum number of decoder-owned arrays smaller than one full decoded
/// weight that are live simultaneously.
///
/// The six one-dimensional code index arrays, the codebook level table, the
/// float32 row scales, and the expanded scale operand. Each is bounded by the
/// larger of the decoded input width and the page's output-row count, which is
/// the shape page admission reserves for them; the input width is at least 32
/// and the level table holds at most 16 entries, so nine such reservations
/// strictly cover every combination, including a synthetic single-row page.
pub const JANGTQ2_DECODE_AUXILIARY_ARRAY_COUNT: usize = 9;

/// Decodes packed codebook rows and applies each row's float16 scale.
///
/// The returned bfloat16 weights remain in the format's stored basis. When
/// `activation_rotation` is Hadamard-32, callers rotate the corresponding
/// projection activations with [`rotate_activations_hadamard32`] before the
/// existing dense expert multiplication. Keeping that rotation on activations
/// avoids materializing a second full expert-weight tensor in the native basis.
pub fn decode_expert_weight_rows(
    runtime: &MlxRuntime,
    packed_code_words: &MlxArray,
    row_scales: &MlxArray,
    bit_width: u32,
) -> Result<MlxArray, MlxRuntimeError> {
    if !codebook::is_supported_bit_width(bit_width) {
        return Err(decode_error(format!(
            "unsupported jangtq2 codebook bit width {bit_width}"
        )));
    }
    if packed_code_words.dtype() != MlxDtype::UInt32 {
        return Err(decode_error(format!(
            "packed code words must use uint32, got {:?}",
            packed_code_words.dtype()
        )));
    }
    if row_scales.dtype() != MlxDtype::Float16 {
        return Err(decode_error(format!(
            "row scales must use float16, got {:?}",
            row_scales.dtype()
        )));
    }

    let packed_shape = packed_code_words.shape();
    let scale_shape = row_scales.shape();
    if packed_shape.len() != 3 || scale_shape.len() != 2 {
        return Err(decode_error(format!(
            "packed tensors must have rank three and row scales rank two, got ranks {} and {}",
            packed_shape.len(),
            scale_shape.len()
        )));
    }
    if packed_shape[0] <= 0
        || packed_shape[1] <= 0
        || packed_shape[2] <= 0
        || packed_shape[0] != scale_shape[0]
        || packed_shape[1] != scale_shape[1]
    {
        return Err(decode_error(format!(
            "packed code shape {packed_shape:?} and row scale shape {scale_shape:?} disagree"
        )));
    }

    let packed_source_shape = packed_shape
        .iter()
        .map(|&dimension| {
            usize::try_from(dimension)
                .map_err(|_| decode_error("packed shape exceeds the host index range".to_owned()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let selected_expert_count = packed_source_shape[0];
    let decoded_weight_shape =
        codebook::decoded_weight_page_shape(&packed_source_shape, bit_width, selected_expert_count)
            .ok_or_else(|| {
                decode_error("packed source shape is not a valid codebook page".to_owned())
            })?;
    let input_width = decoded_weight_shape[2];
    let bit_width_as_usize = bit_width as usize;
    if !hadamard32::is_applicable_to_input_width(input_width) {
        return Err(decode_error(format!(
            "decoded input width {input_width} is not aligned to the jangtq2 packing width"
        )));
    }
    let input_width_i32 = i32::try_from(input_width)
        .map_err(|_| decode_error("decoded input width exceeds the MLX shape range".to_owned()))?;
    let code_index_shape = [input_width_i32];
    let code_mask = (1_u32 << bit_width) - 1;

    let mut low_word_indices = Vec::with_capacity(input_width);
    let mut high_word_indices = Vec::with_capacity(input_width);
    let mut low_bit_shifts = Vec::with_capacity(input_width);
    let mut high_bit_shifts = Vec::with_capacity(input_width);
    let mut crossing_code_masks = Vec::with_capacity(input_width);
    for code_index in 0..input_width {
        let first_code_bit = code_index * bit_width_as_usize;
        let word_index = first_code_bit / u32::BITS as usize;
        let bit_offset = (first_code_bit % u32::BITS as usize) as u32;
        let code_crosses_word_boundary = bit_offset + bit_width > u32::BITS;
        low_word_indices.push(i32::try_from(word_index).map_err(|_| {
            decode_error("packed word index exceeds the MLX index range".to_owned())
        })?);
        high_word_indices.push(
            i32::try_from(if code_crosses_word_boundary {
                word_index + 1
            } else {
                word_index
            })
            .map_err(|_| {
                decode_error("packed word index exceeds the MLX index range".to_owned())
            })?,
        );
        low_bit_shifts.push(bit_offset);
        high_bit_shifts.push(if code_crosses_word_boundary {
            u32::BITS - bit_offset
        } else {
            0
        });
        crossing_code_masks.push(if code_crosses_word_boundary {
            code_mask
        } else {
            0
        });
    }

    let low_word_index_array = MlxArray::from_i32(&low_word_indices, &code_index_shape)?;
    let high_word_index_array = MlxArray::from_i32(&high_word_indices, &code_index_shape)?;
    let low_bit_shift_array = MlxArray::from_u32(&low_bit_shifts, &code_index_shape)?;
    let high_bit_shift_array = MlxArray::from_u32(&high_bit_shifts, &code_index_shape)?;
    let crossing_code_mask_array = MlxArray::from_u32(&crossing_code_masks, &code_index_shape)?;
    let complete_code_mask_array =
        MlxArray::from_u32(&vec![code_mask; input_width], &code_index_shape)?;

    let shifted_low_words = {
        let low_words = runtime.take_axis(packed_code_words, &low_word_index_array, 2)?;
        let shifted_low_words = runtime.right_shift(&low_words, &low_bit_shift_array)?;
        materialize_and_detach_jangtq2_array(runtime, &shifted_low_words)?;
        shifted_low_words
    };
    let shifted_high_words = {
        let high_words = runtime.take_axis(packed_code_words, &high_word_index_array, 2)?;
        let shifted_high_words = runtime.left_shift(&high_words, &high_bit_shift_array)?;
        materialize_and_detach_jangtq2_array(runtime, &shifted_high_words)?;
        shifted_high_words
    };
    let crossing_high_bits = runtime.bitwise_and(&shifted_high_words, &crossing_code_mask_array)?;
    materialize_and_detach_jangtq2_array(runtime, &crossing_high_bits)?;
    drop(shifted_high_words);

    let decoded_code_indices = {
        let combined_codes = runtime.bitwise_or(&shifted_low_words, &crossing_high_bits)?;
        materialize_and_detach_jangtq2_array(runtime, &combined_codes)?;
        drop(shifted_low_words);
        drop(crossing_high_bits);

        let masked_codes = runtime.bitwise_and(&combined_codes, &complete_code_mask_array)?;
        materialize_and_detach_jangtq2_array(runtime, &masked_codes)?;
        drop(combined_codes);

        let decoded_code_indices = runtime.astype(&masked_codes, MlxDtype::Int32)?;
        materialize_and_detach_jangtq2_array(runtime, &decoded_code_indices)?;
        decoded_code_indices
    };

    let codebook_levels = codebook::codebook_levels(bit_width)
        .expect("supported codebook bit widths always provide a level table");
    let codebook_level_count = i32::try_from(codebook_levels.len())
        .map_err(|_| decode_error("codebook level count exceeds the MLX shape range".to_owned()))?;
    let codebook_level_array = MlxArray::from_f32(&codebook_levels, &[codebook_level_count])?;
    let decoded_codebook_levels =
        runtime.take_axis(&codebook_level_array, &decoded_code_indices, 0)?;
    materialize_and_detach_jangtq2_array(runtime, &decoded_codebook_levels)?;
    drop(decoded_code_indices);
    let row_scales_float32 = runtime.astype(row_scales, MlxDtype::Float32)?;
    let expanded_row_scales = runtime.expand_dims(&row_scales_float32, -1)?;
    let decoded_weight_float32 =
        runtime.multiply(&decoded_codebook_levels, &expanded_row_scales)?;
    materialize_and_detach_jangtq2_array(runtime, &decoded_weight_float32)?;
    drop(decoded_codebook_levels);
    let decoded_weight_bfloat16 = runtime.astype(&decoded_weight_float32, MlxDtype::BFloat16)?;
    materialize_and_detach_jangtq2_array(runtime, &decoded_weight_bfloat16)?;
    Ok(decoded_weight_bfloat16)
}

fn materialize_and_detach_jangtq2_array(
    runtime: &MlxRuntime,
    array: &MlxArray,
) -> Result<(), MlxRuntimeError> {
    runtime.evaluate_arrays(&[array])?;
    array
        .detach()
        .map_err(|error| MlxRuntimeError::RuntimeOperation {
            operation: "detach a jangtq2 array",
            description: error.to_string(),
        })
}

/// Applies the blockwise normalized Hadamard-32 to an activation's final axis.
///
/// Reshaping the last axis into 32-wide blocks is essential: a Hadamard over the
/// complete hidden width is a different transform from jangtq2's blockwise one.
pub fn rotate_activations_hadamard32(
    runtime: &MlxRuntime,
    activations: &MlxArray,
) -> Result<MlxArray, MlxRuntimeError> {
    let activation_shape = activations.shape();
    let Some(&input_width_i32) = activation_shape.last() else {
        return Err(rotation_error(
            "activation arrays must have a non-empty shape".to_owned(),
        ));
    };
    if input_width_i32 <= 0 || !hadamard32::is_applicable_to_input_width(input_width_i32 as usize) {
        return Err(rotation_error(format!(
            "activation input width {input_width_i32} is not divisible by {}",
            hadamard32::HADAMARD_BLOCK_WIDTH
        )));
    }
    let mut blocked_activation_shape = activation_shape.clone();
    let Some(last_axis_shape) = blocked_activation_shape.last_mut() else {
        return Err(rotation_error(
            "activation arrays must have a non-empty shape".to_owned(),
        ));
    };
    *last_axis_shape /= hadamard32::HADAMARD_BLOCK_WIDTH as i32;
    blocked_activation_shape.push(hadamard32::HADAMARD_BLOCK_WIDTH as i32);

    let activation_dtype = activations.dtype();
    let activations_float32 = runtime.astype(activations, MlxDtype::Float32)?;
    let blocked_activations = runtime.reshape(&activations_float32, &blocked_activation_shape)?;
    let rotated_blocks = runtime.hadamard_transform(&blocked_activations, None)?;
    let rotated_activation_float32 = runtime.reshape(&rotated_blocks, &activation_shape)?;
    let rotated_activations = runtime.astype(&rotated_activation_float32, activation_dtype)?;
    materialize_and_detach_jangtq2_array(runtime, &rotated_activations)?;
    Ok(rotated_activations)
}

fn decode_error(description: String) -> MlxRuntimeError {
    MlxRuntimeError::RuntimeOperation {
        operation: CODEBOOK_DECODE_OPERATION,
        description,
    }
}

fn rotation_error(description: String) -> MlxRuntimeError {
    MlxRuntimeError::RuntimeOperation {
        operation: HADAMARD_ACTIVATION_ROTATION_OPERATION,
        description,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use astronomical_runtime_integration::MlxMemoryLimits;
    use serial_test::serial;

    fn test_runtime() -> MlxRuntime {
        MlxRuntime::initialize(
            MlxMemoryLimits::new(512 * 1024 * 1024, 512 * 1024 * 1024)
                .expect("the jangtq2 decode test memory limits should be valid"),
        )
        .expect("the direct MLX runtime should initialize")
    }

    fn reference_golden_u32_values(key: &str) -> Vec<u32> {
        serde_json::from_str::<serde_json::Value>(include_str!(
            "tests/golden_reference_vectors.json"
        ))
        .expect("the checked-in jangtq2 golden vectors should be valid JSON")[key]
            .as_array()
            .expect("the requested golden value should be a list")
            .iter()
            .map(|golden_value| {
                u32::try_from(golden_value.as_u64().expect("golden values are unsigned"))
                    .expect("golden packed words fit uint32")
            })
            .collect()
    }

    #[test]
    #[serial]
    fn should_decode_reference_packed_three_bit_codes_on_mlx() {
        let runtime = test_runtime();
        let packed_words = reference_golden_u32_values("pack3_random_words");
        let reference_codes = reference_golden_u32_values("pack3_random_codes");
        let packed_code_words = MlxArray::from_u32(&packed_words, &[1, 1, 3])
            .expect("the golden packed words should form one expert output row");
        let row_scales = runtime
            .array_from_f32(&[1.0], &[1, 1])
            .and_then(|float32_scales| runtime.astype(&float32_scales, MlxDtype::Float16))
            .expect("one float16 codebook row scale should be representable");

        let decoded_weight =
            decode_expert_weight_rows(&runtime, &packed_code_words, &row_scales, 3)
                .expect("the reference 3-bit expert row should decode");
        let decoded_float32_weight = runtime
            .astype(&decoded_weight, MlxDtype::Float32)
            .expect("the decoded bfloat16 row should convert for comparison");
        let decoded_values = decoded_float32_weight
            .to_vec_f32()
            .expect("the decoded float32 comparison row should be readable");

        assert_eq!(decoded_weight.shape(), [1, 1, 32]);
        assert_eq!(decoded_weight.dtype(), MlxDtype::BFloat16);
        for (code_index, (decoded_value, reference_code)) in
            decoded_values.iter().zip(reference_codes).enumerate()
        {
            let reference_level = codebook::codebook_level_value(3, reference_code)
                .expect("reference codes are supported 3-bit values");
            assert!(
                (decoded_value - reference_level).abs() < 0.01,
                "MLX decoded code {code_index} as {decoded_value}, expected {reference_level}"
            );
        }
    }

    #[test]
    #[serial]
    fn should_apply_blockwise_hadamard32_to_mlx_activations() {
        let runtime = test_runtime();
        let activation_values = (0..64)
            .map(|input_index| (input_index as f32 - 17.0) / 11.0)
            .collect::<Vec<_>>();
        let activations = runtime
            .array_from_f32(&activation_values, &[1, 64])
            .expect("the 64-wide activation test tensor should be valid");
        let rotated_activations = rotate_activations_hadamard32(&runtime, &activations)
            .expect("both 32-wide activation blocks should rotate");
        let rotated_values = rotated_activations
            .to_vec_f32()
            .expect("the rotated activation rows should be readable");
        let expected_values = hadamard32::reference_transform_vector(&activation_values)
            .expect("the activation width contains two full Hadamard blocks");

        assert_eq!(rotated_activations.shape(), [1, 64]);
        for (actual_value, expected_value) in rotated_values.iter().zip(expected_values) {
            assert!(
                (actual_value - expected_value).abs() < 1e-4,
                "MLX Hadamard output {actual_value} differs from reference {expected_value}"
            );
        }
    }
}
