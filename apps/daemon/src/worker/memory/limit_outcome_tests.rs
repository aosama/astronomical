//! Unit tests for which memory-limit outcomes may rewrite the relaunch snapshot.

use crate::WorkerControlError;

use super::{MlxMemoryLimitUpdateOutcome, memory_limit_outcome_was_applied};

#[test]
fn should_record_an_applied_memory_ceiling() {
    assert!(memory_limit_outcome_was_applied(&Ok(
        MlxMemoryLimitUpdateOutcome::Applied
    )));
}

#[test]
fn should_refuse_to_record_a_rejected_memory_ceiling() {
    assert!(!memory_limit_outcome_was_applied(&Ok(
        MlxMemoryLimitUpdateOutcome::Rejected
    )));
}

#[test]
fn should_refuse_to_record_when_the_memory_update_fails() {
    assert!(!memory_limit_outcome_was_applied(&Err(
        WorkerControlError::GenerationBusy
    )));
}

#[test]
fn should_refuse_to_record_a_queued_memory_ceiling() {
    assert!(!memory_limit_outcome_was_applied(&Ok(
        MlxMemoryLimitUpdateOutcome::Queued
    )));
}

#[test]
fn should_refuse_to_record_a_restart_required_memory_ceiling() {
    assert!(!memory_limit_outcome_was_applied(&Ok(
        MlxMemoryLimitUpdateOutcome::RestartRequired
    )));
}
