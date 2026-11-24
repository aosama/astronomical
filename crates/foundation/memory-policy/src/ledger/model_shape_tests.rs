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
