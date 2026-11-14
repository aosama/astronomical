use super::{
    CompletedChunkEvidence, PostRecoveryExpertRefill, completed_chunk_measures_remaining_prefill,
};

const MAXIMUM_CHUNK_TOKEN_COUNT: usize = 4_096;
const WORKSPACE_FLOOR_BYTES: u64 = 2_264_924_160;
const MEASURED_WORKSPACE_BYTES: u64 = 6_416_813_132;

/// A full chunk completed with more prompt to come, and its measurement
/// raised the reserve: the case that earns a refill.
fn evidence_of_a_measured_full_chunk() -> CompletedChunkEvidence {
    CompletedChunkEvidence {
        completed_chunk_token_count: MAXIMUM_CHUNK_TOKEN_COUNT,
        remaining_prefill_token_count: 6_666,
        maximum_chunk_token_count: MAXIMUM_CHUNK_TOKEN_COUNT,
        planned_workspace_bytes_at_recovery: WORKSPACE_FLOOR_BYTES,
        planned_workspace_bytes_now: MEASURED_WORKSPACE_BYTES,
    }
}

#[test]
fn should_refill_after_a_full_chunk_whose_measurement_raised_the_reserve() {
    assert!(completed_chunk_measures_remaining_prefill(
        PostRecoveryExpertRefill::NotAttempted,
        evidence_of_a_measured_full_chunk(),
    ));
}

#[test]
fn should_refill_after_a_short_chunk_that_still_covers_what_remains() {
    // A prompt-cache boundary can cut a chunk short. It is still evidence
    // for the rest when nothing larger is left to run.
    for remaining_prefill_token_count in [2_304, 1] {
        assert!(
            completed_chunk_measures_remaining_prefill(
                PostRecoveryExpertRefill::NotAttempted,
                CompletedChunkEvidence {
                    completed_chunk_token_count: 2_304,
                    remaining_prefill_token_count,
                    ..evidence_of_a_measured_full_chunk()
                },
            ),
            "{remaining_prefill_token_count}"
        );
    }
}

#[test]
fn should_not_refill_on_a_chunk_smaller_than_one_still_to_come() {
    assert!(!completed_chunk_measures_remaining_prefill(
        PostRecoveryExpertRefill::NotAttempted,
        CompletedChunkEvidence {
            completed_chunk_token_count: 2_304,
            remaining_prefill_token_count: 8_000,
            ..evidence_of_a_measured_full_chunk()
        },
    ));
    assert!(!completed_chunk_measures_remaining_prefill(
        PostRecoveryExpertRefill::NotAttempted,
        CompletedChunkEvidence {
            completed_chunk_token_count: MAXIMUM_CHUNK_TOKEN_COUNT - 1,
            remaining_prefill_token_count: MAXIMUM_CHUNK_TOKEN_COUNT,
            ..evidence_of_a_measured_full_chunk()
        },
    ));
}

#[test]
fn should_not_refill_when_the_measurement_taught_nothing_new() {
    // Refilling against the reserve that was in force at the failure would
    // rebuild the state that just ran out of memory.
    for planned_workspace_bytes_now in [WORKSPACE_FLOOR_BYTES, WORKSPACE_FLOOR_BYTES - 1, 0] {
        assert!(
            !completed_chunk_measures_remaining_prefill(
                PostRecoveryExpertRefill::NotAttempted,
                CompletedChunkEvidence {
                    planned_workspace_bytes_now,
                    ..evidence_of_a_measured_full_chunk()
                },
            ),
            "{planned_workspace_bytes_now}"
        );
    }
}

#[test]
fn should_not_refill_when_the_prompt_is_finished() {
    // The decode handoff lifts the freeze itself; there is no chunk left
    // for kept layers to help.
    assert!(!completed_chunk_measures_remaining_prefill(
        PostRecoveryExpertRefill::NotAttempted,
        CompletedChunkEvidence {
            remaining_prefill_token_count: 0,
            ..evidence_of_a_measured_full_chunk()
        },
    ));
}

#[test]
fn should_refill_at_most_once_in_a_request() {
    for post_recovery_expert_refill in [
        PostRecoveryExpertRefill::Active,
        PostRecoveryExpertRefill::Spent,
    ] {
        assert!(
            !completed_chunk_measures_remaining_prefill(
                post_recovery_expert_refill,
                evidence_of_a_measured_full_chunk(),
            ),
            "{post_recovery_expert_refill:?}"
        );
    }
}

#[test]
fn should_use_up_an_active_refill_when_a_recovery_follows_it() {
    assert_eq!(
        PostRecoveryExpertRefill::Active.after_recovery(),
        PostRecoveryExpertRefill::Spent
    );
    assert_eq!(
        PostRecoveryExpertRefill::Spent.after_recovery(),
        PostRecoveryExpertRefill::Spent
    );
}

#[test]
fn should_keep_the_one_refill_for_a_request_whose_first_recovery_just_happened() {
    assert_eq!(
        PostRecoveryExpertRefill::NotAttempted.after_recovery(),
        PostRecoveryExpertRefill::NotAttempted
    );
    assert_eq!(
        PostRecoveryExpertRefill::default(),
        PostRecoveryExpertRefill::NotAttempted
    );
}
