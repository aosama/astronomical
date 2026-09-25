//! Hermetic tests for the 3-axis RoPE, compared against the reference frequencies and axis
//! indices for the fixed 16-token joint layout.

use astronomical_model_serving::{QwenImage21Rope, frequencies_for_tests};
use serde_json::Value;

use super::support::{
    ORACLE_TOLERANCE, oracle_document, oracle_f32_values, oracle_index_values, test_image_pad_mask,
    test_img_shapes,
};

/// Reference complex frequencies plus the frame/height/width index table, derived from the
/// diffusers QwenImage21Rope implementation for `test_img_shapes` / `test_image_pad_mask`.
const ROPE_ORACLE_JSON: &str = include_str!("../fixtures/qwen_image_21/rope_oracle.json");

fn rope_oracle() -> Value {
    oracle_document(ROPE_ORACLE_JSON)
}

fn sequence_length(oracle: &Value) -> usize {
    oracle["seq_len"].as_u64().expect("seq_len must exist") as usize
}

fn head_half(oracle: &Value) -> usize {
    oracle["head_half"].as_u64().expect("head_half must exist") as usize
}

/// Compare the computed frequencies against the reference element-wise.
fn assert_matches_oracle(computed: &(Vec<f32>, Vec<f32>), oracle: &Value) {
    let (real, imag) = computed;
    let oracle_real = oracle_f32_values(oracle, "oracle_real");
    let oracle_imag = oracle_f32_values(oracle, "oracle_imag");

    assert_eq!(
        real.len(),
        sequence_length(oracle) * head_half(oracle),
        "real frequency length must equal seq_len * head_half"
    );
    assert_eq!(
        imag.len(),
        real.len(),
        "imaginary and real lengths must match"
    );
    assert_eq!(oracle_real.len(), real.len(), "oracle real length");
    assert_eq!(oracle_imag.len(), imag.len(), "oracle imaginary length");

    for index in 0..real.len() {
        let real_diff = (real[index] - oracle_real[index]).abs();
        let imag_diff = (imag[index] - oracle_imag[index]).abs();
        assert!(
            real_diff <= ORACLE_TOLERANCE,
            "real[{index}]={:?} oracle={:?} (diff {real_diff})",
            real[index],
            oracle_real[index]
        );
        assert!(
            imag_diff <= ORACLE_TOLERANCE,
            "imag[{index}]={:?} oracle={:?} (diff {imag_diff})",
            imag[index],
            oracle_imag[index]
        );
    }
}

#[test]
fn should_match_the_diffusers_oracle_frequencies_exactly() {
    let rope = QwenImage21Rope::default();
    let computed = frequencies_for_tests(&rope, &test_img_shapes(), &test_image_pad_mask());
    assert_matches_oracle(&computed, &rope_oracle());
}

#[test]
fn should_build_the_expected_frame_height_and_width_indices() {
    let rope = QwenImage21Rope::default();
    let oracle = rope_oracle();
    let expected_frame = oracle_index_values(&oracle, "expected_frame");
    let expected_height = oracle_index_values(&oracle, "expected_height");
    let expected_width = oracle_index_values(&oracle, "expected_width");
    let (frame_index, height_index, width_index) =
        rope.construct_indices(&test_img_shapes(), &test_image_pad_mask());

    let expected_length = sequence_length(&oracle);
    assert_eq!(expected_frame.len(), expected_length, "oracle frame length");
    assert_eq!(
        expected_height.len(),
        expected_length,
        "oracle height length"
    );
    assert_eq!(expected_width.len(), expected_length, "oracle width length");
    assert_eq!(frame_index.len(), expected_length);
    assert_eq!(height_index.len(), expected_length);
    assert_eq!(width_index.len(), expected_length);

    for index in 0..expected_length {
        assert_eq!(
            frame_index[index], expected_frame[index],
            "frame index mismatch at {index}"
        );
        assert_eq!(
            height_index[index], expected_height[index],
            "height index mismatch at {index}"
        );
        assert_eq!(
            width_index[index], expected_width[index],
            "width index mismatch at {index}"
        );
    }
}

#[test]
fn should_freeze_the_frame_axis_across_every_image_block() {
    // The two image blocks each share a single frozen frame position; the frame index must be
    // constant within each block and increase across blocks and text.
    let rope = QwenImage21Rope::default();
    let (frame_index, _height_index, _width_index) =
        rope.construct_indices(&test_img_shapes(), &test_image_pad_mask());

    // Block 1 = positions [2, 3, 4, 5] (2x2); block 2 = positions [9..14] (2x3 = 6 tokens).
    let block_one = &frame_index[2..6];
    let block_two = &frame_index[9..15];
    assert!(
        block_one.iter().all(|&value| value == block_one[0]),
        "block 1 must share one frozen frame position: {block_one:?}"
    );
    assert!(
        block_two.iter().all(|&value| value == block_two[0]),
        "block 2 must share one frozen frame position: {block_two:?}"
    );
    assert!(
        block_two[0] > block_one[0],
        "the later block must freeze at a larger frame position: {block_one:?} {block_two:?}"
    );
}

#[test]
fn should_lay_image_blocks_on_a_zero_centered_grid() {
    // The first image block is 2x2: height/width grid indices must be the zero-centered
    // `[-(2 - 1), 2/2)` = [-1, 0] range, so the minimum grid index is -1, never a larger negative.
    let rope = QwenImage21Rope::default();
    let (_frame_index, height_index, width_index) =
        rope.construct_indices(&test_img_shapes(), &test_image_pad_mask());

    let block_one_height = &height_index[2..6];
    let block_one_width = &width_index[2..6];
    assert_eq!(block_one_height.len(), 4);
    assert!(
        block_one_height
            .iter()
            .all(|&value| (value >= -1) && (value <= 0)),
        "2x2 block height grid must be zero-centered [-1, 0]: {block_one_height:?}"
    );
    assert_eq!(
        block_one_height
            .iter()
            .filter(|&&value| value == -1)
            .count(),
        2,
        "the -1 row must appear once per width column"
    );
    assert!(
        block_one_width
            .iter()
            .all(|&value| (value >= -1) && (value <= 0)),
        "2x2 block width grid must be zero-centered [-1, 0]: {block_one_width:?}"
    );
}

#[test]
fn should_support_a_single_condition_block_with_no_trailing_text() {
    // Minimal valid input: one 2x2 block, no text at all, to exercise the block-only path.
    let rope = QwenImage21Rope::default();
    let shapes = [(2, 2)];
    let mask = [false, true, true, true, true];
    let (frame_index, height_index, width_index) = rope.construct_indices(&shapes, &mask);

    assert_eq!(frame_index.len(), 5);
    assert_eq!(height_index.len(), 5);
    assert_eq!(width_index.len(), 5);
    // Text tokens (position 0) advance the shared frame position before the block.
    assert_eq!(frame_index[0], 0);
    assert_eq!(
        &frame_index[1..],
        &[
            frame_index[1],
            frame_index[1],
            frame_index[1],
            frame_index[1]
        ]
    );
}

#[test]
fn should_default_to_the_qwen_image_21_axis_dimensions() {
    let rope = QwenImage21Rope::default();
    let oracle = rope_oracle();
    assert_eq!(rope.axes_dim(), [16, 56, 56]);
    assert_eq!(sequence_length(&oracle), 16);
    assert_eq!(head_half(&oracle), 64);
}
