use super::expert_streaming_summary;
use crate::{ChatGenerationFailureReason, RequestId, WorkerEvent};

const PRIVATE_FAILURE_TEXT: &str = "Shall I compare thee to a summer's day?";

#[test]
fn should_name_a_request_event_with_its_request_id() {
    let failed_event = WorkerEvent::Failed {
        request_id: RequestId::new(42),
        reason: ChatGenerationFailureReason::FatalExecution {
            reason: PRIVATE_FAILURE_TEXT.to_owned(),
        },
    };
    assert_eq!(failed_event.diagnostic_summary(), "failed request_id=42");
}

#[test]
fn should_never_print_the_payload_of_a_request_event() {
    let failed_event = WorkerEvent::Failed {
        request_id: RequestId::new(7),
        reason: ChatGenerationFailureReason::InvalidRequest {
            reason: PRIVATE_FAILURE_TEXT.to_owned(),
        },
    };
    assert!(!failed_event.diagnostic_summary().contains("summer"));
}

#[test]
fn should_print_the_largest_request_id_unchanged() {
    let failed_event = WorkerEvent::Failed {
        request_id: RequestId::new(u64::MAX),
        reason: ChatGenerationFailureReason::FatalExecution {
            reason: String::new(),
        },
    };
    assert_eq!(
        failed_event.diagnostic_summary(),
        format!("failed request_id={}", u64::MAX)
    );
}

#[test]
fn should_name_a_model_event_without_its_failure_text() {
    let swap_failed_event = WorkerEvent::ModelSwapFailed {
        loaded_model_remains_ready: true,
        model_load_failure_reason: PRIVATE_FAILURE_TEXT.to_owned(),
    };
    assert_eq!(swap_failed_event.diagnostic_summary(), "model_swap_failed");
}

#[test]
fn should_name_the_scoped_and_global_prompt_cache_clears_apart() {
    let scoped_clear = WorkerEvent::PromptCacheCleared {
        model_id: Some("ornith-9b".to_owned()),
        blocks_removed: 3,
        bytes_freed: 4096,
    };
    let global_clear = WorkerEvent::PromptCacheCleared {
        model_id: None,
        blocks_removed: 0,
        bytes_freed: 0,
    };
    assert_eq!(
        scoped_clear.diagnostic_summary(),
        "prompt_cache_cleared model_id=Some(\"ornith-9b\")"
    );
    assert_eq!(
        global_clear.diagnostic_summary(),
        "prompt_cache_cleared model_id=None"
    );
}

#[test]
fn should_name_expert_streaming_stats_with_model_and_retained_assignments() {
    assert_eq!(
        expert_streaming_summary("expert_streaming_stats", Some("ornith-35b"), 4_096),
        "expert_streaming_stats model_id=ornith-35b retained_assignments=4096"
    );
}

#[test]
fn should_name_expert_streaming_stats_before_any_model_is_loaded() {
    assert_eq!(
        expert_streaming_summary("expert_streaming_stats", None, 0),
        "expert_streaming_stats model_id=<none> retained_assignments=0"
    );
}
