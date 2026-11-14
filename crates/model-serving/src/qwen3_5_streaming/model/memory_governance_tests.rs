use astronomical_memory_policy::ledger::LedgerPhase;

use super::{
    dense_model_memory_shape, larger_of_what_was_measured, ledger_phase_for,
    sparse_model_memory_shape,
};
use crate::{MemoryPhase, MlxRamBudgetModelGeometry};

#[test]
fn should_govern_a_prompt_forward_as_reading_at_its_token_count() {
    assert_eq!(
        ledger_phase_for(MemoryPhase::Prefill, 4_096),
        Some(LedgerPhase::ReadingPrompt {
            span_token_count: 4_096
        })
    );
}

#[test]
fn should_govern_a_reply_forward_as_writing() {
    assert_eq!(
        ledger_phase_for(MemoryPhase::Decode, 1),
        Some(LedgerPhase::WritingReply)
    );
}

#[test]
fn should_not_govern_phases_that_run_no_measured_forward() {
    assert_eq!(
        ledger_phase_for(MemoryPhase::GenerationPreparation, 1),
        None
    );
    assert_eq!(ledger_phase_for(MemoryPhase::Idle, 0), None);
}

#[test]
fn should_cap_an_absurd_token_count_instead_of_wrapping() {
    assert_eq!(
        ledger_phase_for(MemoryPhase::Prefill, usize::MAX),
        Some(LedgerPhase::ReadingPrompt {
            span_token_count: u32::MAX
        })
    );
}

#[test]
fn should_describe_a_sparse_model_by_its_experts_and_two_layer_slots() {
    let model_shape = sparse_model_memory_shape(MlxRamBudgetModelGeometry {
        model_core_payload_bytes: 2_600,
        complete_expert_payload_bytes: 34_000,
        largest_complete_expert_layer_bytes: 860,
        largest_routed_expert_page_bytes: 27,
        sequence_state_bytes_per_token: 20,
    });
    assert_eq!(model_shape.core_weight_bytes, 2_600);
    assert_eq!(model_shape.streamable_weight_bytes, 34_000);
    assert_eq!(model_shape.reading_prompt_stream_slot_bytes, 1_720);
    assert_eq!(model_shape.writing_reply_stream_slot_bytes, 27);
}

#[test]
fn should_describe_a_dense_model_by_its_decoder_layers_and_one_slot_size() {
    let model_shape = dense_model_memory_shape(900, &[150, 150, 160], 320);
    assert_eq!(model_shape.core_weight_bytes, 900);
    assert_eq!(model_shape.streamable_weight_bytes, 460);
    assert_eq!(model_shape.reading_prompt_stream_slot_bytes, 320);
    assert_eq!(model_shape.writing_reply_stream_slot_bytes, 320);
}

#[test]
fn should_describe_a_dense_model_with_no_streamed_layers_as_having_nothing_to_keep() {
    let model_shape = dense_model_memory_shape(900, &[], 0);
    assert_eq!(model_shape.streamable_weight_bytes, 0);
    assert_eq!(model_shape.reading_prompt_stream_slot_bytes, 0);
}

#[test]
fn should_saturate_instead_of_wrapping_on_absurd_layer_sizes() {
    let model_shape = dense_model_memory_shape(0, &[u64::MAX, 5], 0);
    assert_eq!(model_shape.streamable_weight_bytes, u64::MAX);
}

#[test]
fn should_take_the_larger_of_two_measurements() {
    assert_eq!(larger_of_what_was_measured(Some(3), Some(7)), Some(7));
    assert_eq!(larger_of_what_was_measured(Some(7), Some(3)), Some(7));
}

#[test]
fn should_take_the_only_measurement_there_is() {
    assert_eq!(larger_of_what_was_measured(Some(5), None), Some(5));
    assert_eq!(larger_of_what_was_measured(None, Some(5)), Some(5));
}

#[test]
fn should_report_nothing_measured_when_neither_exists() {
    assert_eq!(larger_of_what_was_measured(None, None), None);
}
