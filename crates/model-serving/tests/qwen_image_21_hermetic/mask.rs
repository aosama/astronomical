//! Hermetic tests for the Qwen-Image-2.1 block-causal attention mask, prefix segmentation, and
//! image-block id construction. These are pure integer bookkeeping with no runtime dependency and
//! are checked against the diffusers reference reimplementation in `mask_fixture.rs`.

use astronomical_model_serving::{build_block_causal_mask, build_image_ids, prefix_segments};

use super::mask_fixture::{
    EXPECTED_IMAGE_IDS, MASK_SEGMENT_COUNT, MASK_SEQ_LEN, ORACLE_MASK, ORACLE_MASK_PADDED,
    PADDED_KEY_INDEX,
};
use super::support::{test_image_pad_mask, test_img_shapes};

const MASK_TOLERANCE: u32 = 0; // exact boolean match against the oracle

fn expected_image_ids() -> Vec<i32> {
    EXPECTED_IMAGE_IDS.to_vec()
}

fn none_case_mask() -> Vec<bool> {
    build_block_causal_mask(&expected_image_ids(), None)
}

fn padded_key_valid() -> Vec<bool> {
    // Only text positions can be invalid keys (image tokens are always valid keys). Positions 8 and
    // 15 are the two trailing text tokens, mirroring a right-padded prompt.
    let mut valid = vec![true; MASK_SEQ_LEN];
    for padded_index in [8usize, PADDED_KEY_INDEX] {
        valid[padded_index] = false;
    }
    valid
}

fn count_mismatches(mask: &[bool], oracle: &[[u8; MASK_SEQ_LEN]]) -> u32 {
    let mut mismatches = 0;
    for row in 0..MASK_SEQ_LEN {
        for col in 0..MASK_SEQ_LEN {
            let expected = oracle[row][col] == 1;
            if mask[row * MASK_SEQ_LEN + col] != expected {
                mismatches += 1;
            }
        }
    }
    mismatches
}

#[test]
fn should_build_the_expected_image_block_ids() {
    let ids = build_image_ids(&test_img_shapes(), &test_image_pad_mask());
    assert_eq!(
        ids,
        expected_image_ids(),
        "image block ids must match the oracle"
    );
}

#[test]
fn should_match_the_diffusers_block_causal_mask() {
    let mask = none_case_mask();
    assert_eq!(mask.len(), MASK_SEQ_LEN * MASK_SEQ_LEN);
    assert_eq!(
        count_mismatches(&mask, &ORACLE_MASK),
        MASK_TOLERANCE,
        "block-causal mask must match the diffusers oracle exactly"
    );
}

#[test]
fn should_match_the_diffusers_padded_key_mask() {
    let mask = build_block_causal_mask(&expected_image_ids(), Some(&padded_key_valid()));
    assert_eq!(mask.len(), MASK_SEQ_LEN * MASK_SEQ_LEN);
    assert_eq!(
        count_mismatches(&mask, &ORACLE_MASK_PADDED),
        MASK_TOLERANCE,
        "padded-key mask must match the diffusers oracle exactly"
    );
}

#[test]
fn should_keep_padded_columns_all_false_and_other_cells_unchanged() {
    let none_case = none_case_mask();
    let padded = build_block_causal_mask(&expected_image_ids(), Some(&padded_key_valid()));

    // Padded keys are excluded as keys: every padded column is entirely False.
    for padded_index in [8usize, PADDED_KEY_INDEX] {
        for row in 0..MASK_SEQ_LEN {
            let kv = row * MASK_SEQ_LEN + padded_index;
            assert!(
                !padded[kv],
                "padded key column {padded_index} must be all False at row {row}"
            );
        }
    }

    // Non-padded cells are unchanged from the no-padding case: padding only removes keys.
    for row in 0..MASK_SEQ_LEN {
        for col in 0..MASK_SEQ_LEN {
            if col == 8 || col == PADDED_KEY_INDEX {
                continue;
            }
            let none_cell = none_case[row * MASK_SEQ_LEN + col];
            let padded_cell = padded[row * MASK_SEQ_LEN + col];
            assert_eq!(
                none_cell, padded_cell,
                "non-padded cell must be unchanged when a key is padded"
            );
        }
    }
}

#[test]
fn should_match_the_diffusers_prefix_segments() {
    let segments = prefix_segments(&expected_image_ids(), MASK_SEQ_LEN);
    assert_eq!(
        segments.len(),
        MASK_SEGMENT_COUNT,
        "prefix segment count must match the oracle"
    );

    // Expected runs: (start, end, is_text).
    let expected = [
        (0usize, 2usize, true),   // leading text
        (2usize, 6usize, false),  // block 0 (2x2)
        (6usize, 9usize, true),   // middle text
        (9usize, 15usize, false), // block 1 (2x3)
        (15usize, 16usize, true), // trailing text
    ];
    assert_eq!(segments.len(), expected.len());
    for (got, want) in segments.iter().zip(expected.iter()) {
        assert_eq!(got, want, "prefix segment must match the oracle");
    }
}

#[test]
fn should_keep_text_rows_causal_and_block_rows_bidirectional() {
    let mask = none_case_mask();
    let id = &expected_image_ids();

    // Text tokens are strictly causal: a text query attends to no later token.
    for q in 0..MASK_SEQ_LEN {
        if id[q] < 0 {
            for kv in (q + 1)..MASK_SEQ_LEN {
                assert!(
                    !mask[q * MASK_SEQ_LEN + kv],
                    "text query {q} must not attend to a later token"
                );
            }
        }
    }

    // Within an image block, tokens attend bidirectionally: earlier tokens attend to later ones.
    for block_id in [0i32, 1i32] {
        let positions: Vec<usize> = (0..MASK_SEQ_LEN).filter(|&t| id[t] == block_id).collect();
        assert!(
            positions.len() >= 2,
            "image block {block_id} must contain at least two tokens"
        );
        for (i, &q) in positions.iter().enumerate() {
            for &kv in &positions[i + 1..] {
                assert!(
                    mask[q * MASK_SEQ_LEN + kv],
                    "block {block_id} token {q} must attend to later block token {kv}"
                );
            }
        }
    }
}
