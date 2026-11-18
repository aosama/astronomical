use super::ModelMemoryShape;
use crate::ledger::phase::LedgerPhase;

const MODEL_SHAPE: ModelMemoryShape = ModelMemoryShape {
    core_weight_bytes: 3_000,
    streamable_weight_bytes: 20_000,
    reading_prompt_stream_slot_bytes: 2_000,
    writing_reply_stream_slot_bytes: 125,
};

#[test]
fn should_need_layer_slots_while_a_prompt_is_read_at_any_span_size() {
    for span_token_count in [1, 4_096] {
        assert_eq!(
            MODEL_SHAPE.stream_slot_bytes_for(LedgerPhase::ReadingPrompt { span_token_count }),
            2_000
        );
    }
}

#[test]
fn should_need_only_the_reply_slot_while_a_reply_is_written() {
    assert_eq!(
        MODEL_SHAPE.stream_slot_bytes_for(LedgerPhase::WritingReply),
        125
    );
}

#[test]
fn should_need_no_slot_while_idle() {
    assert_eq!(MODEL_SHAPE.stream_slot_bytes_for(LedgerPhase::Idle), 0);
}

#[test]
fn should_count_a_companion_models_weights_as_core_weights() {
    let with_drafter = MODEL_SHAPE.with_companion_weight_bytes(1_000);
    assert_eq!(with_drafter.core_weight_bytes, 4_000);
    assert_eq!(
        with_drafter.streamable_weight_bytes, 20_000,
        "the companion is never streamed"
    );
}

#[test]
fn should_leave_the_shape_alone_without_a_companion() {
    assert_eq!(MODEL_SHAPE.with_companion_weight_bytes(0), MODEL_SHAPE);
}

#[test]
fn should_saturate_instead_of_wrapping_on_an_impossible_companion() {
    assert_eq!(
        MODEL_SHAPE
            .with_companion_weight_bytes(u64::MAX)
            .core_weight_bytes,
        u64::MAX
    );
}

#[test]
fn should_need_a_reply_slot_for_every_row_a_step_writes() {
    let drafting = MODEL_SHAPE.with_reply_rows_per_step(4);
    assert_eq!(
        drafting.stream_slot_bytes_for(LedgerPhase::WritingReply),
        500
    );
    assert_eq!(
        drafting.stream_slot_bytes_for(LedgerPhase::ReadingPrompt {
            span_token_count: 1
        }),
        2_000,
        "reading a prompt is unchanged"
    );
}

#[test]
fn should_read_zero_reply_rows_as_the_one_a_reply_always_writes() {
    assert_eq!(MODEL_SHAPE.with_reply_rows_per_step(0), MODEL_SHAPE);
    assert_eq!(MODEL_SHAPE.with_reply_rows_per_step(1), MODEL_SHAPE);
}
