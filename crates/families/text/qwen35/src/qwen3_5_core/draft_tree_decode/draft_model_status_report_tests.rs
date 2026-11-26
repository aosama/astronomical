use ipc_protocol::DraftModelStatus;

use super::{KERNELS_UNAVAILABLE_REASON, failed_load_status, load_outcome_status};
use crate::qwen3_5_core::draft_tree_decode::engine_draft_runtime::{
    Qwen3_5DraftRuntimeLoadOutcome, Qwen3_5SpeculativeDraftLoadingError,
};

#[test]
fn should_report_no_drafter_when_none_pairs() {
    assert_eq!(
        load_outcome_status(&Qwen3_5DraftRuntimeLoadOutcome::NoPairedDrafter),
        DraftModelStatus::None
    );
}

#[test]
fn should_explain_a_gpu_without_the_drafting_kernels() {
    assert_eq!(
        load_outcome_status(&Qwen3_5DraftRuntimeLoadOutcome::KernelsUnavailable),
        DraftModelStatus::Unavailable {
            reason: KERNELS_UNAVAILABLE_REASON.to_owned()
        }
    );
}

#[test]
fn should_name_the_memory_limit_when_there_is_no_room_for_the_drafter() {
    let headroom_error = Qwen3_5SpeculativeDraftLoadingError::InsufficientHeadroom {
        free_bytes: 0,
        required_bytes: 1,
    };
    let DraftModelStatus::Unavailable { reason } = failed_load_status(&headroom_error) else {
        panic!("a drafter that cannot fit is unavailable");
    };
    assert!(reason.contains("memory limit"));
}

#[test]
fn should_never_put_a_local_path_in_a_failure_reason() {
    let unreadable_error = Qwen3_5SpeculativeDraftLoadingError::WeightsUnreadable {
        source: std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "/private/drafter/model.safetensors",
        ),
    };
    let DraftModelStatus::Unavailable { reason } = failed_load_status(&unreadable_error) else {
        panic!("a drafter that cannot be read is unavailable");
    };
    assert!(!reason.contains('/'));
}
