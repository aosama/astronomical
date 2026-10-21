//! Compiled jangtq2 codebook decode graph.
//!
//! The staged decoder evaluates every intermediate array before the next
//! stage records, which costs one graphics-processor synchronization per
//! stage (about eleven per projection page) and re-reads every temporary
//! from memory. Compiling the chain through MLX's compiler fuses the
//! elementwise runs (shifts, bitwise combines, casts, scale multiplication)
//! into single kernels between the three gathers, so one replay decodes a
//! page with one final synchronization and roughly half the memory traffic.
//! The fused elementwise runs never materialize their intermediates, but
//! both gathered word streams outlive the fused bitwise run into the
//! codebook-level gather, so the compiled path coexists four full-width
//! arrays at its peak — the decoder exports that count (four) and page
//! admission reserves it; the decode throughput tier pins the measured
//! peak against the bound so an MLX lifetime change fails loudly.
//!
//! Graphs are compiled per (bit width, decoded input width) with static
//! shapes, because the traced gather shapes depend on the index-table
//! lengths and static-shape compilation sidesteps shapeless tracing's slice
//! limitations. A serving model has a handful of distinct widths, so this
//! compiles a handful of graphs once per inference thread. The index and
//! shift tables and the codebook level table are caller inputs: the jangtq2
//! family owns the format constants, this module owns only compiled
//! execution.

use std::cell::RefCell;
use std::collections::HashMap;
use std::collections::hash_map::Entry as HashMapEntry;
use std::os::raw::c_int;

use crate::MlxArray;
use crate::compile::graph::{
    MlxCompiledGraph, array_from_vector, graph_output_array, set_graph_output,
};
use crate::error::MlxCError;
use crate::raw;
use crate::types::stream::MlxStream;

const COMPILE_JANGTQ2_CODEBOOK_DECODE_OPERATION: &str =
    "compile the MLX jangtq2 codebook decode graph";
const APPLY_JANGTQ2_CODEBOOK_DECODE_OPERATION: &str =
    "apply the compiled MLX jangtq2 codebook decode graph";
const JANGTQ2_DECODE_INPUT_COUNT: usize = 9;

thread_local! {
    /// Serving decodes on the graphics processor run on the inference owner
    /// thread, so a thread-local cache keeps the compiled graphs without
    /// requiring shared ownership of non-Send MLX handles across threads.
    static COMPILED_DECODE_GRAPH_BY_WIDTH: RefCell<HashMap<(u32, usize), MlxCompiledGraph>> =
        RefCell::new(HashMap::new());
}

/// Decodes one packed page or block through the compiled graph.
///
/// `packed_code_words` is `[experts, rows, words]` uint32 and `row_scales`
/// is `[experts, rows]` float16. The six table inputs (`low_word_indices`
/// int32, `high_word_indices` int32, `low_bit_shifts` / `high_bit_shifts` /
/// `crossing_code_masks` / `complete_code_masks` uint32, each `[width]`)
/// and the float32 `codebook_levels` come from the jangtq2 family's
/// golden-vector-pinned construction. The result is the bfloat16 decoded
/// weight in the format's stored basis, identical bit for bit to the staged
/// chain: the same operations run in the same order, and compilation only
/// fuses them.
#[allow(clippy::too_many_arguments)]
pub fn apply_compiled_jangtq2_codebook_decode(
    packed_code_words: &MlxArray,
    row_scales: &MlxArray,
    low_word_indices: &MlxArray,
    high_word_indices: &MlxArray,
    low_bit_shifts: &MlxArray,
    high_bit_shifts: &MlxArray,
    crossing_code_masks: &MlxArray,
    complete_code_masks: &MlxArray,
    codebook_levels: &MlxArray,
    bit_width: u32,
    input_width: usize,
) -> Result<MlxArray, MlxCError> {
    COMPILED_DECODE_GRAPH_BY_WIDTH.with(|compiled_graph_by_width| {
        let mut compiled_graph_by_width = compiled_graph_by_width.borrow_mut();
        let decode_graph = match compiled_graph_by_width.entry((bit_width, input_width)) {
            HashMapEntry::Occupied(occupied_entry) => occupied_entry.into_mut(),
            HashMapEntry::Vacant(vacant_entry) => vacant_entry.insert(
                MlxCompiledGraph::new(
                    build_jangtq2_codebook_decode_graph,
                    COMPILE_JANGTQ2_CODEBOOK_DECODE_OPERATION,
                    false,
                )?,
            ),
        };
        decode_graph.apply(
            &[
                packed_code_words,
                row_scales,
                low_word_indices,
                high_word_indices,
                low_bit_shifts,
                high_bit_shifts,
                crossing_code_masks,
                complete_code_masks,
                codebook_levels,
            ],
            APPLY_JANGTQ2_CODEBOOK_DECODE_OPERATION,
        )
    })
}

unsafe extern "C" fn build_jangtq2_codebook_decode_graph(
    output_vector: *mut raw::mlx_vector_array,
    input_vector: raw::mlx_vector_array,
) -> c_int {
    if output_vector.is_null()
        || unsafe { raw::mlx_vector_array_size(input_vector) } as usize != JANGTQ2_DECODE_INPUT_COUNT
    {
        return 1;
    }
    let packed_code_words = match array_from_vector(input_vector, 0) {
        Ok(packed_code_words) => packed_code_words,
        Err(get_status) => return get_status,
    };
    let row_scales = match array_from_vector(input_vector, 1) {
        Ok(row_scales) => row_scales,
        Err(get_status) => return get_status,
    };
    let low_word_index_array = match array_from_vector(input_vector, 2) {
        Ok(low_word_index_array) => low_word_index_array,
        Err(get_status) => return get_status,
    };
    let high_word_index_array = match array_from_vector(input_vector, 3) {
        Ok(high_word_index_array) => high_word_index_array,
        Err(get_status) => return get_status,
    };
    let low_bit_shift_array = match array_from_vector(input_vector, 4) {
        Ok(low_bit_shift_array) => low_bit_shift_array,
        Err(get_status) => return get_status,
    };
    let high_bit_shift_array = match array_from_vector(input_vector, 5) {
        Ok(high_bit_shift_array) => high_bit_shift_array,
        Err(get_status) => return get_status,
    };
    let crossing_code_mask_array = match array_from_vector(input_vector, 6) {
        Ok(crossing_code_mask_array) => crossing_code_mask_array,
        Err(get_status) => return get_status,
    };
    let complete_code_mask_array = match array_from_vector(input_vector, 7) {
        Ok(complete_code_mask_array) => complete_code_mask_array,
        Err(get_status) => return get_status,
    };
    let codebook_level_array = match array_from_vector(input_vector, 8) {
        Ok(codebook_level_array) => codebook_level_array,
        Err(get_status) => return get_status,
    };
    let gpu_stream = match MlxStream::default_gpu() {
        Ok(gpu_stream) => gpu_stream,
        Err(_) => return 1,
    };
    let gathered_low_words = match graph_output_array(|output_array| {
        // SAFETY: The inputs and stream are live, and the output is uniquely writable.
        unsafe {
            raw::mlx_take_axis(
                output_array,
                packed_code_words.raw(),
                low_word_index_array.raw(),
                2,
                gpu_stream.raw(),
            )
        }
    }) {
        Ok(gathered_low_words) => gathered_low_words,
        Err(build_status) => return build_status,
    };
    let shifted_low_words = match graph_output_array(|output_array| {
        // SAFETY: The inputs and stream are live, and the output is uniquely writable.
        unsafe {
            raw::mlx_right_shift(
                output_array,
                gathered_low_words.raw(),
                low_bit_shift_array.raw(),
                gpu_stream.raw(),
            )
        }
    }) {
        Ok(shifted_low_words) => shifted_low_words,
        Err(build_status) => return build_status,
    };
    let gathered_high_words = match graph_output_array(|output_array| {
        // SAFETY: The inputs and stream are live, and the output is uniquely writable.
        unsafe {
            raw::mlx_take_axis(
                output_array,
                packed_code_words.raw(),
                high_word_index_array.raw(),
                2,
                gpu_stream.raw(),
            )
        }
    }) {
        Ok(gathered_high_words) => gathered_high_words,
        Err(build_status) => return build_status,
    };
    let shifted_high_words = match graph_output_array(|output_array| {
        // SAFETY: The inputs and stream are live, and the output is uniquely writable.
        unsafe {
            raw::mlx_left_shift(
                output_array,
                gathered_high_words.raw(),
                high_bit_shift_array.raw(),
                gpu_stream.raw(),
            )
        }
    }) {
        Ok(shifted_high_words) => shifted_high_words,
        Err(build_status) => return build_status,
    };
    let crossing_high_bits = match graph_output_array(|output_array| {
        // SAFETY: The inputs and stream are live, and the output is uniquely writable.
        unsafe {
            raw::mlx_bitwise_and(
                output_array,
                shifted_high_words.raw(),
                crossing_code_mask_array.raw(),
                gpu_stream.raw(),
            )
        }
    }) {
        Ok(crossing_high_bits) => crossing_high_bits,
        Err(build_status) => return build_status,
    };
    let combined_codes = match graph_output_array(|output_array| {
        // SAFETY: The inputs and stream are live, and the output is uniquely writable.
        unsafe {
            raw::mlx_bitwise_or(
                output_array,
                shifted_low_words.raw(),
                crossing_high_bits.raw(),
                gpu_stream.raw(),
            )
        }
    }) {
        Ok(combined_codes) => combined_codes,
        Err(build_status) => return build_status,
    };
    let masked_codes = match graph_output_array(|output_array| {
        // SAFETY: The inputs and stream are live, and the output is uniquely writable.
        unsafe {
            raw::mlx_bitwise_and(
                output_array,
                combined_codes.raw(),
                complete_code_mask_array.raw(),
                gpu_stream.raw(),
            )
        }
    }) {
        Ok(masked_codes) => masked_codes,
        Err(build_status) => return build_status,
    };
    let decoded_code_indices = match graph_output_array(|output_array| {
        // SAFETY: The input and stream are live, and the output is uniquely writable.
        unsafe {
            raw::mlx_astype(
                output_array,
                masked_codes.raw(),
                raw::mlx_dtype__MLX_INT32,
                gpu_stream.raw(),
            )
        }
    }) {
        Ok(decoded_code_indices) => decoded_code_indices,
        Err(build_status) => return build_status,
    };
    let decoded_codebook_levels = match graph_output_array(|output_array| {
        // SAFETY: The inputs and stream are live, and the output is uniquely writable.
        unsafe {
            raw::mlx_take_axis(
                output_array,
                codebook_level_array.raw(),
                decoded_code_indices.raw(),
                0,
                gpu_stream.raw(),
            )
        }
    }) {
        Ok(decoded_codebook_levels) => decoded_codebook_levels,
        Err(build_status) => return build_status,
    };
    let row_scales_float32 = match graph_output_array(|output_array| {
        // SAFETY: The input and stream are live, and the output is uniquely writable.
        unsafe {
            raw::mlx_astype(
                output_array,
                row_scales.raw(),
                raw::mlx_dtype__MLX_FLOAT32,
                gpu_stream.raw(),
            )
        }
    }) {
        Ok(row_scales_float32) => row_scales_float32,
        Err(build_status) => return build_status,
    };
    let expanded_row_scales = match graph_output_array(|output_array| {
        // SAFETY: The input and stream are live, and the output is uniquely writable.
        unsafe {
            raw::mlx_expand_dims(
                output_array,
                row_scales_float32.raw(),
                -1,
                gpu_stream.raw(),
            )
        }
    }) {
        Ok(expanded_row_scales) => expanded_row_scales,
        Err(build_status) => return build_status,
    };
    let decoded_weight_float32 = match graph_output_array(|output_array| {
        // SAFETY: The inputs and stream are live, and the output is uniquely writable.
        unsafe {
            raw::mlx_multiply(
                output_array,
                decoded_codebook_levels.raw(),
                expanded_row_scales.raw(),
                gpu_stream.raw(),
            )
        }
    }) {
        Ok(decoded_weight_float32) => decoded_weight_float32,
        Err(build_status) => return build_status,
    };
    let decoded_weight_bfloat16 = match graph_output_array(|output_array| {
        // SAFETY: The input and stream are live, and the output is uniquely writable.
        unsafe {
            raw::mlx_astype(
                output_array,
                decoded_weight_float32.raw(),
                raw::mlx_dtype__MLX_BFLOAT16,
                gpu_stream.raw(),
            )
        }
    }) {
        Ok(decoded_weight_bfloat16) => decoded_weight_bfloat16,
        Err(build_status) => return build_status,
    };
    // SAFETY: The output vector is unique and live for this callback.
    unsafe { set_graph_output(output_vector, &decoded_weight_bfloat16) }
}
