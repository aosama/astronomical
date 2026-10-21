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

// Admission-facing decode facts are family format constants shared with the
// un-gated artifact contract and the memory package, so they live in
// `codebook` and are re-exported here for the direct-MLX callers that already
// resolve them through this module.
pub use super::codebook::{
    JANGTQ2_DECODE_AUXILIARY_ARRAY_COUNT, JANGTQ2_DECODE_BLOCK_EXPERT_COUNT,
    MAXIMUM_SIMULTANEOUS_DECODE_TEMPORARY_ARRAY_COUNT, decode_block_expert_count,
    decoded_output_assembly_copy_count,
};

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
    if !hadamard32::is_applicable_to_input_width(input_width) {
        return Err(decode_error(format!(
            "decoded input width {input_width} is not aligned to the jangtq2 packing width"
        )));
    }

    let block_expert_count = decode_block_expert_count(selected_expert_count);
    if block_expert_count >= selected_expert_count {
        // Single-block fast path: the page itself is the block. The per-token
        // routed page (top-K experts) always lands here.
        return decode_expert_weight_page_or_block(
            runtime,
            packed_code_words,
            row_scales,
            bit_width,
            input_width,
        );
    }

    // Complete-layer pages decode in expert blocks so the transient
    // full-width workspace follows the block size rather than the page size
    // (see `JANGTQ2_DECODE_BLOCK_EXPERT_COUNT`). Blocks decode strictly one
    // at a time: each block's staged chain detaches fully before the next
    // block gathers, which is what bounds simultaneous temporaries to the
    // largest single block. The detached block outputs coexist only during
    // the final assembly concatenation, which
    // `decoded_output_assembly_copy_count` prices into the page peak.
    let mut decoded_block_weights =
        Vec::with_capacity(selected_expert_count.div_ceil(block_expert_count));
    for block_start_expert_index in (0..selected_expert_count).step_by(block_expert_count) {
        let block_end_expert_index =
            (block_start_expert_index + block_expert_count).min(selected_expert_count);
        let block_expert_indices = (block_start_expert_index..block_end_expert_index)
            .map(|expert_index| {
                i32::try_from(expert_index).map_err(|_| {
                    decode_error("expert index exceeds the MLX index range".to_owned())
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let block_index_shape = [i32::try_from(block_expert_indices.len()).map_err(|_| {
            decode_error("block expert count exceeds the MLX shape range".to_owned())
        })?];
        let block_index_array = MlxArray::from_i32(&block_expert_indices, &block_index_shape)?;
        // Gathering the block's packed rows and scales materializes one
        // block-sized uint32 copy plus the small float16 scale rows inside
        // the block's own staged evaluation; the page-sized sources stay the
        // caller's loaded payload, exactly as admission expects.
        let block_packed_words = runtime.take_axis(packed_code_words, &block_index_array, 0)?;
        let block_row_scales = runtime.take_axis(row_scales, &block_index_array, 0)?;
        let decoded_block_weight = decode_expert_weight_page_or_block(
            runtime,
            &block_packed_words,
            &block_row_scales,
            bit_width,
            input_width,
        )?;
        drop(block_packed_words);
        drop(block_row_scales);
        decoded_block_weights.push(decoded_block_weight);
    }
    let decoded_block_weight_references = decoded_block_weights.iter().collect::<Vec<_>>();
    let decoded_weight_bfloat16 = runtime.concatenate_axis(&decoded_block_weight_references, 0)?;
    materialize_and_detach_jangtq2_array(runtime, &decoded_weight_bfloat16)?;
    Ok(decoded_weight_bfloat16)
}

/// Per-(bit width, input width) decode tables and constants.
///
/// The six one-dimensional index/shift/mask tables and the codebook level
/// table are pure functions of the width pair, so they are built once per
/// inference thread and shared by every page and block of that width.
struct FamilyDecodeTables {
    bit_width: u32,
    input_width: usize,
    low_word_index_array: MlxArray,
    high_word_index_array: MlxArray,
    low_bit_shift_array: MlxArray,
    high_bit_shift_array: MlxArray,
    crossing_code_mask_array: MlxArray,
    complete_code_mask_array: MlxArray,
    codebook_level_array: MlxArray,
}

thread_local! {
    /// Inference decodes run on the inference owner thread; the table cache
    /// stays thread-local because MLX handles are not shared across threads.
    static FAMILY_DECODE_TABLES_BY_WIDTH: std::cell::RefCell<
        std::collections::HashMap<(u32, usize), FamilyDecodeTables>,
    > = std::cell::RefCell::new(std::collections::HashMap::new());
}

/// Returns the cached decode tables for one width pair, building them first.
fn decode_expert_weight_page_or_block(
    runtime: &MlxRuntime,
    packed_code_words: &MlxArray,
    row_scales: &MlxArray,
    bit_width: u32,
    input_width: usize,
) -> Result<MlxArray, MlxRuntimeError> {
    FAMILY_DECODE_TABLES_BY_WIDTH.with(|decode_tables_by_width| {
        let mut decode_tables_by_width = decode_tables_by_width.borrow_mut();
        let family_decode_tables = match decode_tables_by_width.entry((bit_width, input_width)) {
            std::collections::hash_map::Entry::Occupied(occupied_entry) => {
                occupied_entry.into_mut()
            }
            std::collections::hash_map::Entry::Vacant(vacant_entry) => {
                vacant_entry.insert(build_family_decode_tables(bit_width, input_width)?)
            }
        };
        // The compiled graph replays the exact operation sequence of the
        // previous staged chain — two word gathers, the fused bitwise
        // extraction, the codebook level gather, the row-scale
        // multiplication, and the bfloat16 cast — so the output is identical
        // bit for bit while the elementwise runs fuse into single kernels
        // and the whole page needs one final synchronization instead of one
        // per stage. The fused runs never materialize their intermediates,
        // which is what keeps the exported
        // full-width-temporary contract truthful for
        // admission. The replay is synchronous, so holding the thread-local
        // cache borrow across it cannot re-enter this module.
        let decoded_weight_bfloat16 =
            astronomical_mlx_c_rust::apply_compiled_jangtq2_codebook_decode(
                packed_code_words,
                row_scales,
                &family_decode_tables.low_word_index_array,
                &family_decode_tables.high_word_index_array,
                &family_decode_tables.low_bit_shift_array,
                &family_decode_tables.high_bit_shift_array,
                &family_decode_tables.crossing_code_mask_array,
                &family_decode_tables.complete_code_mask_array,
                &family_decode_tables.codebook_level_array,
                family_decode_tables.bit_width,
                family_decode_tables.input_width,
            )
            .map_err(|apply_error| decode_error(apply_error.to_string()))?;
        // MLX compiles evaluate their outputs eagerly, but the family
        // contract keeps one explicit evaluate-and-detach boundary per
        // decoded page or block so the retained output never pins its source
        // graph in the allocator cache.
        materialize_and_detach_jangtq2_array(runtime, &decoded_weight_bfloat16)?;
        Ok(decoded_weight_bfloat16)
    })
}

fn build_family_decode_tables(
    bit_width: u32,
    input_width: usize,
) -> Result<FamilyDecodeTables, MlxRuntimeError> {
    let bit_width_as_usize = bit_width as usize;
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
    let codebook_levels = codebook::codebook_levels(bit_width)
        .expect("supported codebook bit widths always provide a level table");
    let codebook_level_count = i32::try_from(codebook_levels.len())
        .map_err(|_| decode_error("codebook level count exceeds the MLX shape range".to_owned()))?;
    let input_width_i32 = i32::try_from(input_width)
        .map_err(|_| decode_error("decoded input width exceeds the MLX shape range".to_owned()))?;
    let code_index_shape = [input_width_i32];
    Ok(FamilyDecodeTables {
        bit_width,
        input_width,
        low_word_index_array: MlxArray::from_i32(&low_word_indices, &code_index_shape)?,
        high_word_index_array: MlxArray::from_i32(&high_word_indices, &code_index_shape)?,
        low_bit_shift_array: MlxArray::from_u32(&low_bit_shifts, &code_index_shape)?,
        high_bit_shift_array: MlxArray::from_u32(&high_bit_shifts, &code_index_shape)?,
        crossing_code_mask_array: MlxArray::from_u32(&crossing_code_masks, &code_index_shape)?,
        complete_code_mask_array: MlxArray::from_u32(
            &vec![code_mask; input_width],
            &code_index_shape,
        )?,
        codebook_level_array: MlxArray::from_f32(&codebook_levels, &[codebook_level_count])?,
    })
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
    fn should_decode_a_multi_block_page_identically_to_per_expert_pages() {
        let runtime = test_runtime();
        // Forty experts on a 3-bit, 64-wide page forces two decode blocks
        // (32 + 8), exercising the gather, per-block staging, and assembly
        // concatenation while a per-expert reference decode pins bit-exact
        // parity with the single-block path.
        let expert_count = 40_usize;
        let output_rows = 2_usize;
        let packed_words_per_row = 6_usize;
        let decoded_width = packed_words_per_row * u32::BITS as usize / 3;
        let packed_words_host: Vec<u32> = (0..expert_count * output_rows * packed_words_per_row)
            .map(|word_index| {
                (word_index as u32)
                    .wrapping_mul(2_654_435_761)
                    .rotate_left(7)
            })
            .collect();
        let row_scales_host: Vec<f32> = (0..expert_count * output_rows)
            .map(|scale_index| ((scale_index % 7) as f32 + 1.0) / 4.0)
            .collect();
        let packed_page = MlxArray::from_u32(
            &packed_words_host,
            &[
                expert_count as i32,
                output_rows as i32,
                packed_words_per_row as i32,
            ],
        )
        .expect("the synthetic multi-block page should form a valid tensor");
        let page_scales = runtime
            .array_from_f32(&row_scales_host, &[expert_count as i32, output_rows as i32])
            .and_then(|float32_scales| runtime.astype(&float32_scales, MlxDtype::Float16))
            .expect("the synthetic multi-block row scales should upload");

        let decoded_page = decode_expert_weight_rows(&runtime, &packed_page, &page_scales, 3)
            .expect("the multi-block page should decode");
        assert_eq!(
            decoded_page.shape(),
            [
                expert_count as i32,
                output_rows as i32,
                decoded_width as i32
            ]
        );
        let decoded_page_values = runtime
            .astype(&decoded_page, MlxDtype::Float32)
            .expect("the decoded page should convert for comparison")
            .to_vec_f32()
            .expect("the decoded page values should be readable");

        for expert_index in 0..expert_count {
            let expert_packed_words =
                &packed_words_host[expert_index * output_rows * packed_words_per_row
                    ..(expert_index + 1) * output_rows * packed_words_per_row];
            let expert_packed_page = MlxArray::from_u32(
                expert_packed_words,
                &[1, output_rows as i32, packed_words_per_row as i32],
            )
            .expect("one expert's packed rows should form a valid tensor");
            let expert_row_scales =
                &row_scales_host[expert_index * output_rows..(expert_index + 1) * output_rows];
            let expert_scales = runtime
                .array_from_f32(expert_row_scales, &[1, output_rows as i32])
                .and_then(|float32_scales| runtime.astype(&float32_scales, MlxDtype::Float16))
                .expect("one expert's row scales should upload");
            let decoded_expert =
                decode_expert_weight_rows(&runtime, &expert_packed_page, &expert_scales, 3)
                    .expect("the per-expert reference page should decode");
            let decoded_expert_values = runtime
                .astype(&decoded_expert, MlxDtype::Float32)
                .expect("the per-expert decode should convert for comparison")
                .to_vec_f32()
                .expect("the per-expert decoded values should be readable");
            let page_offset = expert_index * output_rows * decoded_width;
            for (decoded_position, reference_value) in decoded_expert_values.iter().enumerate() {
                assert_eq!(
                    decoded_page_values[page_offset + decoded_position],
                    *reference_value,
                    "expert {expert_index} must decode identically through the blocked path"
                );
            }
        }
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
