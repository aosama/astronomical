//! Unit tests for the pure error-combination helpers of the worker replacement.

use super::combine_publication_errors;
use crate::WorkerControlError;

#[test]
fn should_wrap_publish_error_with_its_failed_cleanup_error() {
    let contained_error = combine_publication_errors(
        WorkerControlError::CandidateConfigurationGenerationMismatch,
        Some(WorkerControlError::GenerationBusy),
    );

    match contained_error {
        WorkerControlError::OperationAndCleanupFailed { operation, cleanup } => {
            assert!(matches!(
                *operation,
                WorkerControlError::CandidateConfigurationGenerationMismatch
            ));
            assert!(matches!(*cleanup, WorkerControlError::GenerationBusy));
        }
        unexpected_error => panic!("unexpected contained error: {unexpected_error}"),
    }
}

#[test]
fn should_keep_publish_error_unchanged_when_cleanup_succeeded() {
    let contained_error = combine_publication_errors(
        WorkerControlError::CandidateConfigurationGenerationMismatch,
        None,
    );

    assert!(matches!(
        contained_error,
        WorkerControlError::CandidateConfigurationGenerationMismatch
    ));
}

#[test]
fn should_nest_an_already_combined_publish_error_inside_the_cleanup_wrapper() {
    let contained_error = combine_publication_errors(
        WorkerControlError::OperationAndCleanupFailed {
            operation: Box::new(WorkerControlError::CandidateConfigurationGenerationMismatch),
            cleanup: Box::new(WorkerControlError::GenerationBusy),
        },
        Some(WorkerControlError::MissingActiveWorker),
    );

    match contained_error {
        WorkerControlError::OperationAndCleanupFailed { operation, cleanup } => {
            assert!(matches!(
                *operation,
                WorkerControlError::OperationAndCleanupFailed { .. }
            ));
            assert!(matches!(*cleanup, WorkerControlError::MissingActiveWorker));
        }
        unexpected_error => panic!("unexpected contained error: {unexpected_error}"),
    }
}
