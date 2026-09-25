//! Hermetic tests for the Qwen-Image-2.1 target-token labelling and `causal_condition` modulation-row
//! selection. These are pure integer bookkeeping with no runtime dependency and are checked against the
//! diffusers reference reimplementation in `modulation_fixture.rs`.

use astronomical_model_serving::{build_target_token_mask, causal_modulation_row_map};

use super::modulation_fixture::{
    EXPECTED_TARGET_MASK, MOD_SEQ_LEN, ORACLE_ROW_MAP_B1, ORACLE_ROW_MAP_B2, ORACLE_ROW_MAP_NONE_B2,
};
use super::support::{test_image_pad_mask, test_img_shapes};

#[test]
fn should_mark_only_the_target_image_tokens() {
    // The target image is the LAST block (the 2x3 at joint positions 9..15); every text token and
    // every condition-image token must stay False. Position 15 is trailing text, so it must NOT be
    // marked even though it sits after the target block.
    let mask = build_target_token_mask(&test_img_shapes(), &test_image_pad_mask());
    assert_eq!(
        mask,
        EXPECTED_TARGET_MASK.to_vec(),
        "target token mask must match the oracle"
    );
}

#[test]
fn should_select_the_t0_row_for_non_target_tokens_at_batch_one() {
    // batch=1: rows [0] is the sample's real timestep row, row [1] is the trailing t=0 row.
    // Only target-image tokens may select row 0; everything else must select row 1.
    let mask = build_target_token_mask(&test_img_shapes(), &test_image_pad_mask());
    let rows = causal_modulation_row_map(Some(&mask), 1, MOD_SEQ_LEN);
    assert_eq!(rows.len(), 1, "row map must have one sample row");
    assert_eq!(
        rows[0],
        ORACLE_ROW_MAP_B1[0].to_vec(),
        "batch=1 row map must match the oracle"
    );
}

#[test]
fn should_select_per_sample_rows_for_multi_batch() {
    // batch=2: rows [0] and [1] are the two samples' real timestep rows, row [2] is the trailing t=0
    // row shared by every non-target token across both samples.
    let mask = build_target_token_mask(&test_img_shapes(), &test_image_pad_mask());
    let rows = causal_modulation_row_map(Some(&mask), 2, MOD_SEQ_LEN);
    assert_eq!(rows.len(), 2, "row map must have two sample rows");
    for sample in 0..2 {
        assert_eq!(
            rows[sample],
            ORACLE_ROW_MAP_B2[sample].to_vec(),
            "batch=2 row {sample} must match the oracle"
        );
    }
}

#[test]
fn should_use_own_sample_row_for_every_token_without_target_mask() {
    // None disables the causal_condition split: every token uses its own sample's real row, and no
    // token falls back to the t=0 row.
    let rows = causal_modulation_row_map(None, 2, MOD_SEQ_LEN);
    assert_eq!(rows.len(), 2, "row map must have two sample rows");
    for sample in 0..2 {
        assert_eq!(
            rows[sample],
            ORACLE_ROW_MAP_NONE_B2[sample].to_vec(),
            "None-case row {sample} must match the oracle"
        );
    }
}

#[test]
fn should_never_reference_a_row_beyond_the_causal_condition_tensor() {
    // The causal_condition tensor carries batch_size + 1 rows. Every selected index must stay within
    // [0, batch_size]; selecting batch_size + 1 or higher would read past the tensor.
    let mask = build_target_token_mask(&test_img_shapes(), &test_image_pad_mask());
    for batch_size in [1usize, 2usize, 4usize] {
        let rows = causal_modulation_row_map(Some(&mask), batch_size, MOD_SEQ_LEN);
        let max_allowed = batch_size; // batch_size rows [0..batch) plus the trailing t=0 row at batch_size
        for sample in 0..batch_size {
            for token in 0..MOD_SEQ_LEN {
                assert!(
                    rows[sample][token] <= max_allowed,
                    "row index {} exceeds the causal_condition tensor bound {max_allowed} (batch={batch_size}, sample={sample}, token={token})",
                    rows[sample][token]
                );
            }
        }
    }
}
