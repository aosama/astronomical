//! The kinds of problem a report can describe, each with the plain words the
//! person reads on screen and the stable name the developers group by, and
//! the one policy for which failures are worth a report at all.

use ipc_protocol::ChatGenerationFailureReason;
use serde::Serialize;

use crate::library::DownloadJobPublicErrorCode;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ProblemKind {
    /// A model could not be loaded because the Mac lacked the memory.
    ModelLoadNotEnoughMemory,
    /// A model could not be loaded for another reason.
    ModelLoadFailed,
    /// A reply stopped because the graphics chip ran out of memory.
    ReplyOutOfMemory,
    /// The engine's worker stopped after a fatal error during a reply.
    EngineStoppedUnexpectedly,
    /// Downloaded files did not match the catalog's record of them.
    DownloadChecksumMismatch,
    /// A catalog model could not be fetched from Hugging Face at all.
    CatalogModelUnavailable,
    /// A model downloaded from the catalog could not be used.
    DownloadedModelNotUsable,
}

impl ProblemKind {
    /// The words shown after "Problem:" in the report the person reads.
    pub(crate) const fn plain_description(self) -> &'static str {
        match self {
            Self::ModelLoadNotEnoughMemory => "model load failed, not enough memory",
            Self::ModelLoadFailed => "model load failed",
            Self::ReplyOutOfMemory => "reply stopped, graphics memory ran out",
            Self::EngineStoppedUnexpectedly => "engine stopped unexpectedly",
            Self::DownloadChecksumMismatch => "downloaded files did not match the catalog",
            Self::CatalogModelUnavailable => "catalog model could not be downloaded",
            Self::DownloadedModelNotUsable => "downloaded model could not be used",
        }
    }

    /// The kind for a failed model load. The worker's reason text is read
    /// only to tell a memory shortfall from other failures; none of it
    /// enters the report.
    pub(crate) fn for_model_load_failure(model_load_failure_reason: &str) -> Self {
        if model_load_failure_reason
            .to_ascii_lowercase()
            .contains("memory")
        {
            Self::ModelLoadNotEnoughMemory
        } else {
            Self::ModelLoadFailed
        }
    }

    /// The kind for a reply that failed, or `None` when the failure is the
    /// request's own (too long, malformed, engine busy) rather than a fault.
    pub(crate) const fn for_reply_failure(
        failure_reason: &ChatGenerationFailureReason,
    ) -> Option<Self> {
        match failure_reason {
            ChatGenerationFailureReason::OutOfGraphicsMemory => Some(Self::ReplyOutOfMemory),
            ChatGenerationFailureReason::FatalExecution { .. } => {
                Some(Self::EngineStoppedUnexpectedly)
            }
            ChatGenerationFailureReason::ContextLengthExceeded { .. }
            | ChatGenerationFailureReason::InvalidRequest { .. }
            | ChatGenerationFailureReason::EngineBusy
            | ChatGenerationFailureReason::MalformedModelOutput => None,
        }
    }

    /// The kind for a failed catalog download, or `None` for everyday causes
    /// (no connection, a full disk, a model already present) that are not a
    /// fault the developers can fix.
    pub(crate) const fn for_download_failure(
        error_code: DownloadJobPublicErrorCode,
    ) -> Option<Self> {
        match error_code {
            DownloadJobPublicErrorCode::ChecksumMismatch => Some(Self::DownloadChecksumMismatch),
            DownloadJobPublicErrorCode::CatalogEntryNotFound
            | DownloadJobPublicErrorCode::ModelNotPublic
            | DownloadJobPublicErrorCode::DownloadGated => Some(Self::CatalogModelUnavailable),
            DownloadJobPublicErrorCode::ModelNotExecutable => Some(Self::DownloadedModelNotUsable),
            DownloadJobPublicErrorCode::LibraryBusy
            | DownloadJobPublicErrorCode::InsufficientDisk
            | DownloadJobPublicErrorCode::DownloadFailed
            | DownloadJobPublicErrorCode::ModelAlreadyPresent => None,
        }
    }
}

#[cfg(test)]
#[path = "problem_kind_tests.rs"]
mod problem_kind_tests;
