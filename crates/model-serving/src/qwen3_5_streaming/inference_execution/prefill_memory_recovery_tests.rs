use super::super::prompt_prefill_errors::PromptPrefillChunkAttemptError;
use super::{
    LayerMajorDecline, MAXIMUM_PREFILL_MEMORY_RECOVERY_ATTEMPTS_PER_REQUEST,
    layer_major_decline_for, should_attempt_prefill_memory_recovery,
};

#[test]
fn should_attempt_recovery_for_the_first_text_prefill_memory_failure() {
    assert!(should_attempt_prefill_memory_recovery(0, false));
}

#[test]
fn should_not_attempt_recovery_once_the_request_used_its_recovery() {
    assert!(!should_attempt_prefill_memory_recovery(
        MAXIMUM_PREFILL_MEMORY_RECOVERY_ATTEMPTS_PER_REQUEST,
        false
    ));
}

#[test]
fn should_not_attempt_recovery_for_image_requests() {
    assert!(!should_attempt_prefill_memory_recovery(0, true));
}

#[test]
fn should_not_attempt_recovery_when_the_attempt_counter_is_saturated() {
    assert!(!should_attempt_prefill_memory_recovery(u8::MAX, false));
}

#[test]
fn should_continue_in_chunks_without_a_recovery_when_admission_refused_a_group() {
    let admission_refusal = PromptPrefillChunkAttemptError::AdaptiveMemoryLimitExceeded {
        decision: crate::PagedForwardAdmissionDecision::Reject {
            unresolved_shortfall_bytes: 1,
        },
        reason: "group too large".to_owned(),
    };
    assert_eq!(
        layer_major_decline_for(&admission_refusal),
        Some(LayerMajorDecline::NotAdmitted)
    );
}

#[test]
fn should_rebuild_state_when_a_group_ran_out_of_memory() {
    let active_memory_overrun = PromptPrefillChunkAttemptError::ActiveMemoryLimitExceeded {
        active_memory_bytes: 3,
        attempted_allocation_bytes: 2,
        allowed_active_memory_bytes: 4,
    };
    let graphics_memory_exhausted =
        PromptPrefillChunkAttemptError::GraphicsProcessorMemoryExhausted {
            reason: "exhausted".to_owned(),
        };
    assert_eq!(
        layer_major_decline_for(&active_memory_overrun),
        Some(LayerMajorDecline::RanOutOfMemory)
    );
    assert_eq!(
        layer_major_decline_for(&graphics_memory_exhausted),
        Some(LayerMajorDecline::RanOutOfMemory)
    );
}

#[test]
fn should_end_the_request_on_an_engine_failure_during_a_group() {
    let engine_failure =
        PromptPrefillChunkAttemptError::Engine(crate::InferenceEngineError::EngineBusy);
    assert_eq!(layer_major_decline_for(&engine_failure), None);
}
