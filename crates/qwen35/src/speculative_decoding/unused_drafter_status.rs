//! What status says about a drafter an engine cannot use, shared by the
//! engines so the wording stays one. Family-neutral: the caller passes its
//! target's shape.

use std::path::Path;

use ipc_protocol::DraftModelStatus;

use super::drafter_discovery::{DrafterPairing, paired_drafter_directory};

/// Reason given when a drafter pairs but the model streams from the SSD.
/// Starts at the cause: every surface already says a drafter is installed.
pub const STREAMING_REASON: &str = "the model is reading its weights from the SSD, and drafting runs only when the whole model is in memory";

/// Reason given when the library could not be searched for a drafter.
pub const LIBRARY_UNSEARCHABLE_REASON: &str = "the model library could not be searched for one";

/// The status for a model the SSD-streaming engine loaded, which never
/// drafts: a drafter that would pair is reported as unused, with the reason,
/// so a person knows what they would gain from more memory.
#[must_use]
pub fn streaming_load_status(
    target_model_directory: &Path,
    pairing: &DrafterPairing,
) -> DraftModelStatus {
    match paired_drafter_directory(target_model_directory, pairing) {
        Ok(Some(_)) => DraftModelStatus::Unavailable {
            reason: STREAMING_REASON.to_owned(),
        },
        Ok(None) => DraftModelStatus::None,
        Err(_) => DraftModelStatus::Unavailable {
            reason: LIBRARY_UNSEARCHABLE_REASON.to_owned(),
        },
    }
}

#[cfg(test)]
#[path = "unused_drafter_status_tests.rs"]
mod tests;
