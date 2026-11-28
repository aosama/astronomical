use memory_policy::ledger::LedgerPhase;

use super::{larger_of_what_was_measured, ledger_phase_for};
use crate::MemoryPhase;

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
