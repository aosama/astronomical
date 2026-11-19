use mlx_bridge::MlxDtype;

use super::{
    FusedProjectionDtypes, PackedRowGeometry, is_shared_activation_row, packed_row_geometry,
    validate_fused_projection_dtypes,
};

/// A 2-bit gate projection of a 256-expert layer: 512 output rows over a
/// 3,072-wide input, which packs to 3,072 · 2 / 32 = 192 words per row.
const EXPERT_COUNT: i32 = 256;
const OUTPUT_ROWS: i32 = 512;
const TWO_BIT_WORDS_PER_ROW: i32 = 192;
const INPUT_WIDTH: i32 = 3_072;

fn supported_dtypes() -> FusedProjectionDtypes {
    FusedProjectionDtypes {
        bit_width: 2,
        packed_code_word_dtype: MlxDtype::UInt32,
        row_scale_dtype: MlxDtype::Float16,
        expert_index_dtype: MlxDtype::UInt32,
        activation_dtype: MlxDtype::BFloat16,
    }
}

fn refusal_text<Accepted: std::fmt::Debug>(
    validation_outcome: Result<Accepted, astronomical_runtime_integration::MlxRuntimeError>,
) -> String {
    validation_outcome
        .expect_err("the input must be refused")
        .to_string()
}

#[test]
fn should_accept_the_dtypes_the_metal_source_reads() {
    assert!(validate_fused_projection_dtypes(&supported_dtypes()).is_ok());
}

#[test]
fn should_refuse_each_dtype_the_kernel_would_reinterpret() {
    let float32_scales = FusedProjectionDtypes {
        row_scale_dtype: MlxDtype::Float32,
        ..supported_dtypes()
    };
    assert!(refusal_text(validate_fused_projection_dtypes(&float32_scales)).contains("float16"));
    let signed_codes = FusedProjectionDtypes {
        packed_code_word_dtype: MlxDtype::Int32,
        ..supported_dtypes()
    };
    assert!(refusal_text(validate_fused_projection_dtypes(&signed_codes)).contains("uint32"));
    let five_bit_codebook = FusedProjectionDtypes {
        bit_width: 5,
        ..supported_dtypes()
    };
    assert!(
        refusal_text(validate_fused_projection_dtypes(&five_bit_codebook)).contains("bit width 5")
    );
}

#[test]
fn should_derive_the_input_width_from_the_packed_row() {
    assert_eq!(
        packed_row_geometry(
            &[EXPERT_COUNT, OUTPUT_ROWS, TWO_BIT_WORDS_PER_ROW],
            &[EXPERT_COUNT, OUTPUT_ROWS],
            2,
        )
        .expect("a well-formed 2-bit page"),
        PackedRowGeometry {
            output_rows: OUTPUT_ROWS,
            words_per_row: TWO_BIT_WORDS_PER_ROW,
            input_width: INPUT_WIDTH,
        }
    );
}

#[test]
fn should_refuse_scales_for_a_different_expert_page() {
    let refusal = refusal_text(packed_row_geometry(
        &[EXPERT_COUNT, OUTPUT_ROWS, TWO_BIT_WORDS_PER_ROW],
        &[EXPERT_COUNT, OUTPUT_ROWS - 1],
        2,
    ));
    assert!(refusal.contains("disagree"), "{refusal}");
}

#[test]
fn should_refuse_a_packed_row_that_is_not_whole_code_groups() {
    let refusal = refusal_text(packed_row_geometry(
        &[EXPERT_COUNT, OUTPUT_ROWS, 191],
        &[EXPERT_COUNT, OUTPUT_ROWS],
        2,
    ));
    assert!(refusal.contains("32-code groups"), "{refusal}");
}

#[test]
fn should_refuse_a_packed_row_whose_width_overflows_instead_of_wrapping() {
    let largest_even_word_count = i32::MAX - 1;
    let refusal = refusal_text(packed_row_geometry(
        &[1, 1, largest_even_word_count],
        &[1, 1],
        2,
    ));
    assert!(refusal.contains("exceeds the MLX shape range"), "{refusal}");
}

#[test]
fn should_refuse_packed_tensors_of_the_wrong_rank() {
    let refusal = refusal_text(packed_row_geometry(
        &[OUTPUT_ROWS, TWO_BIT_WORDS_PER_ROW],
        &[OUTPUT_ROWS],
        2,
    ));
    assert!(refusal.contains("rank three"), "{refusal}");
}

#[test]
fn should_share_one_activation_row_across_every_decode_assignment() {
    assert_eq!(
        is_shared_activation_row(&[1, INPUT_WIDTH], INPUT_WIDTH, 8).ok(),
        Some(true)
    );
    assert_eq!(
        is_shared_activation_row(&[8, INPUT_WIDTH], INPUT_WIDTH, 8).ok(),
        Some(false)
    );
}

#[test]
fn should_refuse_activations_that_match_neither_one_row_nor_every_assignment() {
    let refusal = refusal_text(is_shared_activation_row(&[4, INPUT_WIDTH], INPUT_WIDTH, 8));
    assert!(refusal.contains("4 rows"), "{refusal}");
    let empty_route = refusal_text(is_shared_activation_row(&[1, INPUT_WIDTH], INPUT_WIDTH, 0));
    assert!(
        empty_route.contains("at least one assignment"),
        "{empty_route}"
    );
    let narrow_activations = refusal_text(is_shared_activation_row(
        &[1, INPUT_WIDTH / 2],
        INPUT_WIDTH,
        8,
    ));
    assert!(
        narrow_activations.contains("[1 | assignments"),
        "{narrow_activations}"
    );
}

#[test]
fn should_refuse_a_packed_page_with_a_zero_or_negative_dimension() {
    for (packed_shape, scale_shape) in [
        ([0, OUTPUT_ROWS, TWO_BIT_WORDS_PER_ROW], [0, OUTPUT_ROWS]),
        ([EXPERT_COUNT, 0, TWO_BIT_WORDS_PER_ROW], [EXPERT_COUNT, 0]),
        ([EXPERT_COUNT, OUTPUT_ROWS, -2], [EXPERT_COUNT, OUTPUT_ROWS]),
    ] {
        let refusal = refusal_text(packed_row_geometry(&packed_shape, &scale_shape, 2));
        assert!(refusal.contains("disagree"), "{packed_shape:?}: {refusal}");
    }
}
