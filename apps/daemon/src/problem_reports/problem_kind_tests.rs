use ipc_protocol::ChatGenerationFailureReason;

use super::ProblemKind;
use crate::library::DownloadJobPublicErrorCode;

#[test]
fn should_tell_a_memory_shortfall_from_other_load_failures_by_code() {
    assert_eq!(
        ProblemKind::for_model_load_failure("not_enough_memory"),
        ProblemKind::ModelLoadNotEnoughMemory
    );
    assert_eq!(
        ProblemKind::for_model_load_failure("memory_admission_failed"),
        ProblemKind::ModelLoadNotEnoughMemory
    );
    assert_eq!(
        ProblemKind::for_model_load_failure("model_files_invalid"),
        ProblemKind::ModelLoadFailed
    );
    assert_eq!(
        ProblemKind::for_model_load_failure(""),
        ProblemKind::ModelLoadFailed
    );
}

#[test]
fn should_report_only_reply_failures_that_are_faults() {
    assert_eq!(
        ProblemKind::for_reply_failure(&ChatGenerationFailureReason::OutOfGraphicsMemory),
        Some(ProblemKind::ReplyOutOfMemory)
    );
    assert_eq!(
        ProblemKind::for_reply_failure(&ChatGenerationFailureReason::FatalExecution {
            reason: "worker crashed".to_owned()
        }),
        Some(ProblemKind::EngineStoppedUnexpectedly)
    );
    for request_failure in [
        ChatGenerationFailureReason::EngineBusy,
        ChatGenerationFailureReason::MalformedModelOutput,
        ChatGenerationFailureReason::InvalidRequest {
            reason: "bad".to_owned(),
        },
    ] {
        assert_eq!(ProblemKind::for_reply_failure(&request_failure), None);
    }
}

#[test]
fn should_skip_everyday_download_failures_and_report_catalog_faults() {
    assert_eq!(
        ProblemKind::for_download_failure(DownloadJobPublicErrorCode::ChecksumMismatch),
        Some(ProblemKind::DownloadChecksumMismatch)
    );
    assert_eq!(
        ProblemKind::for_download_failure(DownloadJobPublicErrorCode::DownloadGated),
        Some(ProblemKind::CatalogModelUnavailable)
    );
    assert_eq!(
        ProblemKind::for_download_failure(DownloadJobPublicErrorCode::ModelNotExecutable),
        Some(ProblemKind::DownloadedModelNotUsable)
    );
    for everyday_failure in [
        DownloadJobPublicErrorCode::DownloadFailed,
        DownloadJobPublicErrorCode::InsufficientDisk,
        DownloadJobPublicErrorCode::LibraryBusy,
        DownloadJobPublicErrorCode::ModelAlreadyPresent,
    ] {
        assert_eq!(ProblemKind::for_download_failure(everyday_failure), None);
    }
}
