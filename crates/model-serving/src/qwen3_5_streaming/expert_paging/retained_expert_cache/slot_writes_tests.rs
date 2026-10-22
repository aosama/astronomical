//! Direct-MLX contracts for warm-table slot writes of packed jangtq2 codebook
//! pages: the table keeps the packed codes and row scales (never a decoded
//! copy), a row write moves exactly one expert's codes and scales, and the
//! writer refuses representation and bit-width mismatches.

use super::retained_expert_cache_tests::test_runtime;
use super::slot_writes::{create_warm_table_weights, write_expert_into_slot};
use super::*;
use crate::qwen3_5_core::model_math::decoder_layer_weights::Qwen3_5AffineWeights;
use astronomical_mlx_c_rust::{MlxArray, MlxDtype};
use astronomical_runtime_integration::MlxRuntime;
use serial_test::serial;

const OUTPUT_ROWS: usize = 2;
/// One 32-code group at 2 bits: two uint32 words per row.
const WORDS_PER_ROW: usize = 2;
const TWO_BIT_WIDTH: u32 = 2;
const THREE_BIT_WIDTH: u32 = 3;

/// A packed page whose words encode their own expert and row so a row read
/// back from a slot identifies exactly where it came from.
fn packed_codebook_page(
    runtime: &MlxRuntime,
    expert_count: usize,
    codebook_bit_width: u32,
) -> Qwen3_5PagedExpertWeights {
    let projection = |projection_tag: u32| {
        let packed_words: Vec<u32> = (0..expert_count * OUTPUT_ROWS * WORDS_PER_ROW)
            .map(|word_index| projection_tag * 1_000_000 + word_index as u32)
            .collect();
        let packed_code_words = MlxArray::from_u32(
            &packed_words,
            &[
                expert_count as i32,
                OUTPUT_ROWS as i32,
                WORDS_PER_ROW as i32,
            ],
        )
        .expect("the packed test page should upload");
        let row_scales_host: Vec<f32> = (0..expert_count * OUTPUT_ROWS)
            .map(|row_index| (projection_tag as f32) + (row_index as f32) / 8.0)
            .collect();
        let row_scales = runtime
            .astype(
                &runtime
                    .array_from_f32(&row_scales_host, &[expert_count as i32, OUTPUT_ROWS as i32])
                    .expect("the row scales should upload"),
                MlxDtype::Float16,
            )
            .expect("the row scales should convert to float16");
        Qwen3_5AffineWeights::Jangtq2Codebook {
            packed_code_words,
            row_scales,
            codebook_bit_width,
        }
    };
    Qwen3_5PagedExpertWeights {
        gate_projection: projection(1),
        up_projection: projection(2),
        down_projection: projection(3),
        gate_activation_rotation:
            astronomical_jangtq2_qwen35::configuration::Jangtq2CodebookRotation::Hadamard32,
        up_activation_rotation:
            astronomical_jangtq2_qwen35::configuration::Jangtq2CodebookRotation::Hadamard32,
        down_activation_rotation:
            astronomical_jangtq2_qwen35::configuration::Jangtq2CodebookRotation::Hadamard32,
    }
}

fn packed_words_and_scales(
    runtime: &MlxRuntime,
    projection: &Qwen3_5AffineWeights,
) -> (Vec<u32>, Vec<f32>, u32) {
    match projection {
        Qwen3_5AffineWeights::Jangtq2Codebook {
            packed_code_words,
            row_scales,
            codebook_bit_width,
        } => (
            packed_code_words
                .to_vec_u32()
                .expect("the packed words should read back"),
            runtime
                .array_to_vec_f32(
                    &runtime
                        .astype(row_scales, MlxDtype::Float32)
                        .expect("the row scales should widen"),
                )
                .expect("the row scales should read back"),
            *codebook_bit_width,
        ),
        other => panic!("the table must stay a packed codebook page, got {other:?}"),
    }
}

#[test]
#[serial]
fn should_pad_a_packed_codebook_page_to_warm_capacity_without_decoding_it() {
    let runtime = test_runtime();
    let streamed = packed_codebook_page(&runtime, 2, TWO_BIT_WIDTH);
    let table = create_warm_table_weights(&runtime, &streamed, 5)
        .expect("the warm table should be created");
    let (packed_words, row_scales, codebook_bit_width) =
        packed_words_and_scales(&runtime, &table.gate_projection);
    assert_eq!(codebook_bit_width, TWO_BIT_WIDTH);
    assert_eq!(packed_words.len(), 5 * OUTPUT_ROWS * WORDS_PER_ROW);
    assert!(packed_words.iter().all(|word| *word == 0));
    assert_eq!(row_scales.len(), 5 * OUTPUT_ROWS);
    assert!(row_scales.iter().all(|scale| *scale == 0.0));
    // Packed storage is the whole point: five slots of a 2×2-word page are
    // 5 × 2 × 2 × 4 bytes of codes plus 5 × 2 × 2 bytes of float16 scales.
    let packed_bytes_per_slot = (OUTPUT_ROWS * WORDS_PER_ROW * 4 + OUTPUT_ROWS * 2) as u64;
    assert_eq!(
        crate::expert_paging::ExpertWeightPage::resident_payload_byte_count(&table),
        5 * 3 * packed_bytes_per_slot
    );
}

#[test]
#[serial]
fn should_write_one_expert_row_of_codes_and_scales_into_a_slot() {
    let runtime = test_runtime();
    let streamed = packed_codebook_page(&runtime, 3, TWO_BIT_WIDTH);
    let mut table = create_warm_table_weights(&runtime, &streamed, 4)
        .expect("the warm table should be created");
    write_expert_into_slot(&runtime, &mut table, &streamed, 2, 1)
        .expect("the row write should succeed");
    let (table_words, table_scales, _) = packed_words_and_scales(&runtime, &table.down_projection);
    let (source_words, source_scales, _) =
        packed_words_and_scales(&runtime, &streamed.down_projection);
    let row_word_count = OUTPUT_ROWS * WORDS_PER_ROW;
    assert_eq!(
        &table_words[row_word_count..2 * row_word_count],
        &source_words[2 * row_word_count..3 * row_word_count],
        "slot 1 must hold expert 2's packed words"
    );
    assert_eq!(
        &table_scales[OUTPUT_ROWS..2 * OUTPUT_ROWS],
        &source_scales[2 * OUTPUT_ROWS..3 * OUTPUT_ROWS],
        "slot 1 must hold expert 2's row scales"
    );
    assert!(
        table_words[..row_word_count].iter().all(|word| *word == 0)
            && table_words[2 * row_word_count..]
                .iter()
                .all(|word| *word == 0),
        "no other slot may change"
    );
}

#[test]
#[serial]
fn should_reject_writing_a_packed_codebook_row_into_a_native_table() {
    let runtime = test_runtime();
    let native = super::retained_expert_cache_tests::streamed_weights_for_expert_count(&runtime, 2);
    let mut native_table = create_warm_table_weights(&runtime, &native, 2)
        .expect("the native warm table should be created");
    let packed = packed_codebook_page(&runtime, 2, TWO_BIT_WIDTH);
    let error = write_expert_into_slot(&runtime, &mut native_table, &packed, 0, 0)
        .expect_err("a packed row cannot land in a bfloat16 table");
    assert!(error.to_string().contains("mismatched quantization"));
}

#[test]
#[serial]
fn should_reject_packed_codebook_rows_of_a_different_bit_width() {
    let runtime = test_runtime();
    let two_bit = packed_codebook_page(&runtime, 2, TWO_BIT_WIDTH);
    let mut two_bit_table = create_warm_table_weights(&runtime, &two_bit, 2)
        .expect("the two-bit warm table should be created");
    let three_bit = packed_codebook_page(&runtime, 2, THREE_BIT_WIDTH);
    let error = write_expert_into_slot(&runtime, &mut two_bit_table, &three_bit, 0, 0)
        .expect_err("rows of another bit width would decode with the wrong level table");
    assert!(error.to_string().contains("mismatched quantization"));
}
