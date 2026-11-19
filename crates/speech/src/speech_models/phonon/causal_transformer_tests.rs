//! Feeding a transformer in pieces through its attention history must give
//! what one call over the whole sequence gives: frame by frame for the flow
//! model, block by block (with old keys forgotten) for the Mimi decoder.

use astronomical_runtime_integration::MlxRuntime;
use mlx_bridge::MlxArray;
use serial_test::serial;

use super::{CausalTransformer, TransformerShape};
use crate::speech_models::checkpoint_tensors::CheckpointTensors;
use crate::speech_models::speech_test_support::{
    assert_close, checkpoint_from_arrays, initialize_test_runtime, read_back, shakespeare_values,
    upload,
};

const WIDTH: i32 = 8;
const FEED_FORWARD_WIDTH: i32 = 16;
const SEQUENCE_LENGTH: i32 = 7;
/// The same arithmetic in different matrix shapes runs on different GPU
/// kernels, which round float32 differently: measured differences are about
/// 5e-4 on values near 1, even on the first row, which both paths compute
/// identically. A wrong mask, position or trimmed key moves values by tenths.
const KERNEL_ROUNDING_TOLERANCE: f32 = 2e-3;

const PREFIX: &str = "transformer";

fn layer_tensor(suffix: &str) -> String {
    format!("{PREFIX}.layers.0.{suffix}")
}

/// Weights from the play, scaled down so activations stay moderate.
fn shakespeare_weight(runtime: &MlxRuntime, byte_offset: usize, shape: &[i32]) -> MlxArray {
    let value_count = shape.iter().product::<i32>() as usize;
    let scaled: Vec<f32> = shakespeare_values(value_count, byte_offset)
        .iter()
        .map(|value| value * 0.2)
        .collect();
    upload(runtime, &scaled, shape)
}

/// One layer with weights from the play, small enough to compare exactly.
fn shakespeare_transformer(
    runtime: &MlxRuntime,
    attention_window: Option<usize>,
) -> CausalTransformer {
    let named_arrays = vec![
        (
            layer_tensor("norm1.weight"),
            upload(runtime, &[1.0; WIDTH as usize], &[WIDTH]),
        ),
        (
            layer_tensor("norm1.bias"),
            upload(runtime, &[0.0; WIDTH as usize], &[WIDTH]),
        ),
        (
            layer_tensor("norm2.weight"),
            upload(runtime, &[1.0; WIDTH as usize], &[WIDTH]),
        ),
        (
            layer_tensor("norm2.bias"),
            upload(runtime, &[0.0; WIDTH as usize], &[WIDTH]),
        ),
        (
            layer_tensor("self_attn.in_proj.weight"),
            shakespeare_weight(runtime, 0, &[3 * WIDTH, WIDTH]),
        ),
        (
            layer_tensor("self_attn.out_proj.weight"),
            shakespeare_weight(runtime, 500, &[WIDTH, WIDTH]),
        ),
        (
            layer_tensor("linear1.weight"),
            shakespeare_weight(runtime, 900, &[FEED_FORWARD_WIDTH, WIDTH]),
        ),
        (
            layer_tensor("linear2.weight"),
            shakespeare_weight(runtime, 1300, &[WIDTH, FEED_FORWARD_WIDTH]),
        ),
        (
            layer_tensor("layer_scale_1.scale"),
            upload(runtime, &[0.5; WIDTH as usize], &[WIDTH]),
        ),
        (
            layer_tensor("layer_scale_2.scale"),
            upload(runtime, &[0.5; WIDTH as usize], &[WIDTH]),
        ),
    ];
    let tensors = checkpoint_from_arrays(runtime, &named_arrays);
    let checkpoint = CheckpointTensors {
        runtime,
        tensors: &tensors,
    };
    let shape = TransformerShape {
        prefix: PREFIX,
        layer_count: 1,
        head_count: 2,
        rotary_max_period: 10_000.0,
        attention_window,
        has_branch_scales: true,
    };
    CausalTransformer::load(&checkpoint, &shape).expect("the transformer loads")
}

fn sequence(runtime: &MlxRuntime) -> MlxArray {
    upload(
        runtime,
        &shakespeare_values((SEQUENCE_LENGTH * WIDTH) as usize, 2000),
        &[SEQUENCE_LENGTH, WIDTH],
    )
}

/// The transformer over the sequence in pieces of `piece_length`.
fn in_pieces(runtime: &MlxRuntime, transformer: &CausalTransformer, piece_length: i32) -> Vec<f32> {
    let whole_sequence = sequence(runtime);
    let mut history = transformer.empty_history();
    let mut outputs = Vec::new();
    let mut piece_start = 0;
    while piece_start < SEQUENCE_LENGTH {
        let piece_end = (piece_start + piece_length).min(SEQUENCE_LENGTH);
        let piece = runtime
            .slice(
                &whole_sequence,
                &[piece_start, 0],
                &[piece_end, WIDTH],
                &[1, 1],
            )
            .expect("slice");
        outputs.extend(read_back(
            runtime,
            &transformer
                .forward(runtime, &piece, &mut history)
                .expect("forward"),
        ));
        piece_start = piece_end;
    }
    outputs
}

#[test]
#[serial]
fn should_give_the_same_states_frame_by_frame_as_all_at_once() {
    let runtime = initialize_test_runtime();
    let transformer = shakespeare_transformer(&runtime, None);
    assert_close(
        &in_pieces(&runtime, &transformer, 1),
        &in_pieces(&runtime, &transformer, SEQUENCE_LENGTH),
        KERNEL_ROUNDING_TOLERANCE,
    );
}

// Boundary: a block length that does not divide the sequence, longer than the window.
#[test]
#[serial]
fn should_give_the_same_states_in_windowed_blocks_as_all_at_once() {
    let runtime = initialize_test_runtime();
    let transformer = shakespeare_transformer(&runtime, Some(3));
    let all_at_once = in_pieces(&runtime, &transformer, SEQUENCE_LENGTH);
    assert_close(
        &in_pieces(&runtime, &transformer, 3),
        &all_at_once,
        KERNEL_ROUNDING_TOLERANCE,
    );
    assert_close(
        &in_pieces(&runtime, &transformer, 2),
        &all_at_once,
        KERNEL_ROUNDING_TOLERANCE,
    );
    assert_close(
        &in_pieces(&runtime, &transformer, 1),
        &all_at_once,
        KERNEL_ROUNDING_TOLERANCE,
    );
}

#[test]
#[serial]
fn should_leave_the_original_history_unchanged_when_a_branch_grows() {
    let runtime = initialize_test_runtime();
    let transformer = shakespeare_transformer(&runtime, None);
    let whole_sequence = sequence(&runtime);
    let first_position = runtime
        .slice(&whole_sequence, &[0, 0], &[1, WIDTH], &[1, 1])
        .expect("slice");
    let mut history = transformer.empty_history();
    transformer
        .forward(&runtime, &first_position, &mut history)
        .expect("forward");
    let mut branch = history.branch().expect("branch");
    transformer
        .forward(&runtime, &first_position, &mut branch)
        .expect("forward");
    assert_eq!(history.absolute_position_count(), 1);
    assert_eq!(history.stored_count(), 1);
    assert_eq!(branch.stored_count(), 2);
}
