//! Direct-MLX contracts for selecting an expert subset out of a stacked page:
//! the subset keeps each representation's arrays and metadata, follows the
//! requested slot order, and rejects slots outside the page.

use super::Qwen3_5PagedExpertWeights;
use crate::expert_paging::ExpertWeightPage;
use crate::qwen3_5_core::model_math::decoder_layer_weights::Qwen3_5AffineWeights;
use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};
use mlx_c_rust::{MlxArray, MlxDtype};
use serial_test::serial;

const PAGE_EXPERT_COUNT: usize = 4;
const OUTPUT_ROWS: usize = 2;
const WORDS_PER_ROW: usize = 2;

fn test_runtime() -> MlxRuntime {
    MlxRuntime::initialize(
        MlxMemoryLimits::new(512 * 1024 * 1024, 512 * 1024 * 1024)
            .expect("the expert subset test memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize")
}

/// A packed codebook page whose every word encodes its own expert and row.
fn packed_codebook_page(runtime: &MlxRuntime) -> Qwen3_5PagedExpertWeights {
    Qwen3_5PagedExpertWeights {
        gate_projection: tagged_codebook_projection(runtime, 1),
        up_projection: tagged_codebook_projection(runtime, 2),
        down_projection: tagged_codebook_projection(runtime, 3),
        gate_activation_rotation:
            astronomical_jangtq2_qwen35::configuration::Jangtq2CodebookRotation::Hadamard32,
        up_activation_rotation:
            astronomical_jangtq2_qwen35::configuration::Jangtq2CodebookRotation::Hadamard32,
        down_activation_rotation:
            astronomical_jangtq2_qwen35::configuration::Jangtq2CodebookRotation::Hadamard32,
    }
}

fn packed_words_of(projection: &Qwen3_5AffineWeights) -> (Vec<u32>, u32) {
    match projection {
        Qwen3_5AffineWeights::Jangtq2Codebook {
            packed_code_words,
            codebook_bit_width,
            ..
        } => (
            packed_code_words
                .to_vec_u32()
                .expect("the packed words should read back"),
            *codebook_bit_width,
        ),
        other => panic!("the subset must keep the packed codebook representation, got {other:?}"),
    }
}

#[test]
#[serial]
fn should_select_experts_in_the_requested_order_and_keep_the_representation() {
    let runtime = test_runtime();
    let page = packed_codebook_page(&runtime);
    let subset = page
        .select_experts(&runtime, &[3, 1])
        .expect("two experts should select");
    let (subset_words, codebook_bit_width) = packed_words_of(&subset.down_projection);
    let (page_words, _) = packed_words_of(&page.down_projection);
    let expert_word_count = OUTPUT_ROWS * WORDS_PER_ROW;
    assert_eq!(codebook_bit_width, 2);
    assert_eq!(subset_words.len(), 2 * expert_word_count);
    assert_eq!(
        &subset_words[..expert_word_count],
        &page_words[3 * expert_word_count..4 * expert_word_count],
        "subset slot 0 must be page expert 3"
    );
    assert_eq!(
        &subset_words[expert_word_count..],
        &page_words[expert_word_count..2 * expert_word_count],
        "subset slot 1 must be page expert 1"
    );
    assert_eq!(subset.gate_projection.expert_count(), 2);
    assert_eq!(
        subset.resident_payload_byte_count(),
        page.resident_payload_byte_count() / 2,
        "two of four experts hold half the page's bytes"
    );
    assert_eq!(
        subset.gate_activation_rotation,
        astronomical_jangtq2_qwen35::configuration::Jangtq2CodebookRotation::Hadamard32
    );
}

#[test]
#[serial]
fn should_select_a_single_expert_and_the_whole_page() {
    let runtime = test_runtime();
    let page = packed_codebook_page(&runtime);
    let single = page
        .select_experts(&runtime, &[0])
        .expect("one expert should select");
    assert_eq!(single.up_projection.expert_count(), 1);
    let whole = page
        .select_experts(&runtime, &[0, 1, 2, 3])
        .expect("every expert should select");
    assert_eq!(
        packed_words_of(&whole.gate_projection).0,
        packed_words_of(&page.gate_projection).0
    );
}

#[test]
#[serial]
fn should_reject_slots_outside_the_page_and_empty_selections() {
    let runtime = test_runtime();
    let page = packed_codebook_page(&runtime);
    let outside = page
        .select_experts(&runtime, &[1, PAGE_EXPERT_COUNT])
        .expect_err("a slot past the expert axis cannot be gathered");
    assert!(outside.to_string().contains("outside a page"));
    let empty = page
        .select_experts(&runtime, &[])
        .expect_err("an empty subset has no page to build");
    assert!(empty.to_string().contains("at least one page slot"));
}

/// One packed codebook projection whose every word encodes its own expert
/// and row, tagged so the gate, up, and down pages of a test page carry
/// distinct, predictable values.
fn tagged_codebook_projection(runtime: &MlxRuntime, projection_tag: u32) -> Qwen3_5AffineWeights {
    let packed_words: Vec<u32> = (0..PAGE_EXPERT_COUNT * OUTPUT_ROWS * WORDS_PER_ROW)
        .map(|word_index| projection_tag * 1_000 + word_index as u32)
        .collect();
    let packed_code_words = MlxArray::from_u32(
        &packed_words,
        &[
            PAGE_EXPERT_COUNT as i32,
            OUTPUT_ROWS as i32,
            WORDS_PER_ROW as i32,
        ],
    )
    .expect("the packed test page should upload");
    let row_scales_host: Vec<f32> = (0..PAGE_EXPERT_COUNT * OUTPUT_ROWS)
        .map(|row_index| projection_tag as f32 + row_index as f32 / 8.0)
        .collect();
    let row_scales = runtime
        .astype(
            &runtime
                .array_from_f32(
                    &row_scales_host,
                    &[PAGE_EXPERT_COUNT as i32, OUTPUT_ROWS as i32],
                )
                .expect("the row scales should upload"),
            MlxDtype::Float16,
        )
        .expect("the row scales should convert to float16");
    Qwen3_5AffineWeights::Jangtq2Codebook {
        packed_code_words,
        row_scales,
        codebook_bit_width: 2,
    }
}
