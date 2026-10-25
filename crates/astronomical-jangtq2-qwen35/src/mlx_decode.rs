//! MLX operations for decoding one bounded jangtq2 expert page.
//!
//! The CPU format contract owns bit extraction and cubic codebook semantics;
//! this module mirrors those rules on the graphics processor so selected
//! expert pages can be decoded without a full weight-stack copy through host
//! memory. The decode is one custom Metal launch per page: it reads the
//! packed words once and writes the bfloat16 weights once, with no
//! intermediate arrays.
//!
//! The previous decoder expressed the same arithmetic as an MLX graph of
//! gathers, shifts, and masks. Even compiled, that graph materialized four
//! full-width uint32 temporaries per projection (about 1.5 GB per
//! complete-layer projection, 9 GB before expert blocking), ran at about
//! 10 GB/s of decoded output (146 ms per complete-layer projection), and
//! forced page admission to reserve the temporaries and block-assembly copies
//! before any expert could load. The custom kernel is bandwidth-bound on the
//! output write, needs no workspace, and lets admission price only the packed
//! source plus the decoded output.
//!
//! Serving uses this decoder only for multi-token prefill; decode tokens
//! multiply straight from the packed codes through
//! `fused_gathered_projection` and never materialize weights.

use std::cell::RefCell;
use std::collections::HashMap;

use astronomical_runtime_integration::{MlxRuntime, MlxRuntimeError};
use mlx_c_rust::{
    MlxArray, MlxDtype, MlxMetalKernel, MlxMetalKernelOutput, MlxMetalKernelTemplateArgument,
};

use super::codebook;
use super::hadamard32;

mod kernel_sources;

const CODEBOOK_DECODE_OPERATION: &str = "decode jangtq2 codebook expert rows";
const HADAMARD_ACTIVATION_ROTATION_OPERATION: &str = "rotate jangtq2 expert activations";
/// Threads per threadgroup for the one-dimensional launch. Every page has a
/// row count times group count that is a multiple of 256 on the served
/// geometry, and the launcher dispatches exact thread counts regardless.
const THREADGROUP_SIZE: i32 = 256;

/// Decodes packed codebook rows and applies each row's float16 scale.
///
/// The returned bfloat16 weights remain in the format's stored basis. When
/// `activation_rotation` is Hadamard-32, callers rotate the corresponding
/// projection activations with [`rotate_activations_hadamard32`] before the
/// existing dense expert multiplication. Keeping that rotation on activations
/// avoids materializing a second full expert-weight tensor in the native basis.
///
/// The decoded page is evaluated and detached before it is returned so the
/// retained output never pins its packed source in the allocator cache.
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
    let input_width_i32 = i32::try_from(input_width)
        .map_err(|_| decode_error("decoded input width exceeds the MLX shape range".to_owned()))?;
    let total_row_count = packed_shape[0]
        .checked_mul(packed_shape[1])
        .ok_or_else(|| decode_error("page row count exceeds the MLX grid range".to_owned()))?;
    let thread_count = total_row_count
        .checked_mul(input_width_i32 / hadamard32::HADAMARD_BLOCK_WIDTH as i32)
        .ok_or_else(|| decode_error("page decode grid exceeds the MLX grid range".to_owned()))?;
    let bits_i32 = i32::try_from(bit_width)
        .map_err(|_| decode_error("bit width exceeds the template range".to_owned()))?;

    let decoded_weight_bfloat16 = PAGE_DECODE_KERNEL.with(|page_decode_kernel| {
        let mut page_decode_kernel = page_decode_kernel.borrow_mut();
        let page_decode_kernel = match page_decode_kernel.as_mut() {
            Some(page_decode_kernel) => page_decode_kernel,
            None => page_decode_kernel.insert(PageDecodeKernel::new()?),
        };
        let codebook_level_table = page_decode_kernel.codebook_level_table(bit_width)?;
        let template_arguments = [
            MlxMetalKernelTemplateArgument::Dtype {
                name: "OutputT",
                dtype: MlxDtype::BFloat16,
            },
            MlxMetalKernelTemplateArgument::Integer {
                name: "bits",
                integer_template_argument: bits_i32,
            },
            MlxMetalKernelTemplateArgument::Integer {
                name: "input_width",
                integer_template_argument: input_width_i32,
            },
            MlxMetalKernelTemplateArgument::Integer {
                name: "words_per_row",
                integer_template_argument: packed_shape[2],
            },
        ];
        let mut outputs = runtime
            .apply_metal_kernel(
                &page_decode_kernel.kernel,
                &[packed_code_words, row_scales, &codebook_level_table],
                &[MlxMetalKernelOutput::new(
                    vec![packed_shape[0], packed_shape[1], input_width_i32],
                    MlxDtype::BFloat16,
                )],
                [thread_count, 1, 1],
                [THREADGROUP_SIZE, 1, 1],
                &template_arguments,
            )
            .map_err(MlxRuntimeError::from)?;
        outputs
            .pop()
            .ok_or_else(|| decode_error("the page decode kernel returned no output".to_owned()))
    })?;
    materialize_and_detach_jangtq2_array(runtime, &decoded_weight_bfloat16)?;
    Ok(decoded_weight_bfloat16)
}

/// The compiled page decode kernel plus the per-width codebook level tables
/// it reads, built once per inference thread.
struct PageDecodeKernel {
    kernel: MlxMetalKernel,
    codebook_level_tables_by_bit_width: HashMap<u32, MlxArray>,
}

thread_local! {
    /// Inference decodes run on the inference owner thread; the kernel owner
    /// stays thread-local because MLX handles are not shared across threads.
    static PAGE_DECODE_KERNEL: RefCell<Option<PageDecodeKernel>> = const { RefCell::new(None) };
}

impl PageDecodeKernel {
    fn new() -> Result<Self, MlxRuntimeError> {
        let kernel = MlxMetalKernel::new(
            "jangtq2_page_decode",
            &["packed_weights", "row_scales", "codebook_levels"],
            &["decoded"],
            kernel_sources::PAGE_DECODE_KERNEL_SOURCE,
        )
        .map_err(MlxRuntimeError::from)?;
        Ok(Self {
            kernel,
            codebook_level_tables_by_bit_width: HashMap::new(),
        })
    }

    /// Returns the level table for one width, uploading it on first use so
    /// a complete-layer decode of three projections never repeats the copy.
    fn codebook_level_table(&mut self, bit_width: u32) -> Result<MlxArray, MlxRuntimeError> {
        if let Some(level_table) = self.codebook_level_tables_by_bit_width.get(&bit_width) {
            return level_table.retain().map_err(MlxRuntimeError::from);
        }
        let levels = codebook::codebook_levels(bit_width).ok_or_else(|| {
            decode_error(format!(
                "unsupported jangtq2 codebook bit width {bit_width}"
            ))
        })?;
        let level_count = i32::try_from(levels.len())
            .map_err(|_| decode_error("level table exceeds the MLX shape range".to_owned()))?;
        let level_table = MlxArray::from_f32(&levels, &[level_count])?;
        level_table.evaluate()?;
        let retained = level_table.retain()?;
        self.codebook_level_tables_by_bit_width
            .insert(bit_width, level_table);
        Ok(retained)
    }
}

fn materialize_and_detach_jangtq2_array(
    runtime: &MlxRuntime,
    array: &MlxArray,
) -> Result<(), MlxRuntimeError> {
    runtime.evaluate_arrays(&[array])?;
    astronomical_mlx_c_fork::detach_array(array).map_err(|error| {
        MlxRuntimeError::RuntimeOperation {
            operation: "detach a jangtq2 array",
            description: error.to_string(),
        }
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
    // The rotation stays lazy on purpose: the projection kernel consumes it in
    // the same graph, and forcing an evaluation here drained the whole pending
    // decode graph once per expert projection (~109 synchronous waits per
    // token on the 122B bundle, about 1 ms each), which dominated decode time.
    runtime
        .astype(&rotated_activation_float32, activation_dtype)
        .map_err(MlxRuntimeError::from)
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
    fn should_decode_a_multi_expert_page_identically_to_per_expert_pages() {
        let runtime = test_runtime();
        // Forty experts on a 3-bit, 64-wide page: every row of the page is
        // decoded by its own threads, and a per-expert reference decode pins
        // bit-exact parity between a large grid and a one-expert grid, so
        // row indexing inside the kernel cannot drift with the page size.
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
                    "expert {expert_index} must decode identically inside the whole page"
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
