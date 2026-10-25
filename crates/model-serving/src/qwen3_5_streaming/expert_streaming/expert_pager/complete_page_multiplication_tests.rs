//! Unit contracts for choosing the on-device complete-layer multiply: affine
//! and bfloat16 pages always take it, codebook pages only for a single decode
//! token with the fused kernel.

use super::Qwen3_5PagedExpertWeights;
use crate::qwen3_5_core::model_math::decoder_layer_weights::Qwen3_5AffineWeights;
use astronomical_jangtq2_qwen35::configuration::Jangtq2CodebookRotation;
use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};
use mlx_c_rust::MlxArray;
use serial_test::serial;

const SINGLE_DECODE_TOKEN_COUNT: i32 = 1;
const PREFILL_CHUNK_TOKEN_COUNT: i32 = 2_048;

fn initialize_test_runtime() -> MlxRuntime {
    MlxRuntime::initialize(
        MlxMemoryLimits::new(512 * 1024 * 1024, 512 * 1024 * 1024)
            .expect("the test memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize")
}

fn placeholder_array() -> MlxArray {
    MlxArray::from_u32(&[0, 1, 2, 3], &[1, 2, 2]).expect("a placeholder page array should upload")
}

fn affine_quantized_projection() -> Qwen3_5AffineWeights {
    Qwen3_5AffineWeights::Quantized {
        packed_weight: placeholder_array(),
        quantization_scales: placeholder_array(),
        quantization_biases: placeholder_array(),
        quantization_bits: 4,
        quantization_group_size: 64,
    }
}

fn native_bfloat16_projection() -> Qwen3_5AffineWeights {
    Qwen3_5AffineWeights::NativeBfloat16 {
        weight: placeholder_array(),
    }
}

fn codebook_projection() -> Qwen3_5AffineWeights {
    Qwen3_5AffineWeights::Jangtq2Codebook {
        packed_code_words: placeholder_array(),
        row_scales: placeholder_array(),
        codebook_bit_width: 2,
    }
}

fn page_of(
    gate_projection: Qwen3_5AffineWeights,
    up_projection: Qwen3_5AffineWeights,
    down_projection: Qwen3_5AffineWeights,
) -> Qwen3_5PagedExpertWeights {
    Qwen3_5PagedExpertWeights {
        gate_projection,
        up_projection,
        down_projection,
        gate_activation_rotation: Jangtq2CodebookRotation::NoRotation,
        up_activation_rotation: Jangtq2CodebookRotation::NoRotation,
        down_activation_rotation: Jangtq2CodebookRotation::NoRotation,
    }
}

#[test]
#[serial]
fn should_multiply_a_complete_affine_page_for_decode_and_prefill_without_the_fused_kernel() {
    let _runtime = initialize_test_runtime();
    let affine_page = page_of(
        affine_quantized_projection(),
        affine_quantized_projection(),
        affine_quantized_projection(),
    );
    assert!(
        affine_page
            .multiplies_complete_page_without_transient_decode(SINGLE_DECODE_TOKEN_COUNT, false)
    );
    assert!(
        affine_page
            .multiplies_complete_page_without_transient_decode(PREFILL_CHUNK_TOKEN_COUNT, false)
    );
}

#[test]
#[serial]
fn should_multiply_a_complete_native_bfloat16_page_for_a_prefill_chunk() {
    let _runtime = initialize_test_runtime();
    let bfloat16_page = page_of(
        native_bfloat16_projection(),
        native_bfloat16_projection(),
        native_bfloat16_projection(),
    );
    assert!(
        bfloat16_page
            .multiplies_complete_page_without_transient_decode(PREFILL_CHUNK_TOKEN_COUNT, true)
    );
}

#[test]
#[serial]
fn should_multiply_a_complete_codebook_page_only_for_one_decode_token_with_the_fused_kernel() {
    let _runtime = initialize_test_runtime();
    let codebook_page = page_of(
        codebook_projection(),
        codebook_projection(),
        codebook_projection(),
    );
    assert!(
        codebook_page
            .multiplies_complete_page_without_transient_decode(SINGLE_DECODE_TOKEN_COUNT, true)
    );
}

#[test]
#[serial]
fn should_refuse_a_complete_codebook_page_without_the_fused_kernel() {
    let _runtime = initialize_test_runtime();
    let codebook_page = page_of(
        codebook_projection(),
        codebook_projection(),
        codebook_projection(),
    );
    assert!(
        !codebook_page
            .multiplies_complete_page_without_transient_decode(SINGLE_DECODE_TOKEN_COUNT, false)
    );
}

#[test]
#[serial]
fn should_refuse_a_complete_codebook_page_for_a_two_token_chunk_even_with_the_fused_kernel() {
    let _runtime = initialize_test_runtime();
    let codebook_page = page_of(
        codebook_projection(),
        codebook_projection(),
        codebook_projection(),
    );
    assert!(!codebook_page.multiplies_complete_page_without_transient_decode(2, true));
}

#[test]
#[serial]
fn should_refuse_a_mixed_page_when_any_projection_needs_a_transient_decode() {
    let _runtime = initialize_test_runtime();
    let mixed_page = page_of(
        affine_quantized_projection(),
        affine_quantized_projection(),
        codebook_projection(),
    );
    assert!(
        !mixed_page
            .multiplies_complete_page_without_transient_decode(PREFILL_CHUNK_TOKEN_COUNT, true)
    );
}
