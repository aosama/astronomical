//! Selects exactly one complete MTP source before optional weight binding.

use crate::{Qwen3_5MtpPairingCompatibility, ValidatedQwen3_5StandaloneMtpArtifact};

/// Complete source selected for one loaded Qwen target.
pub enum Qwen3_5MtpSourceSelection {
    TargetLocal,
    Standalone {
        artifact: ValidatedQwen3_5StandaloneMtpArtifact,
        compatibility: Qwen3_5MtpPairingCompatibility,
    },
    TargetOnly {
        reason: Qwen3_5MtpSourceUnavailableReason,
        drafter_model_id: String,
        drafter_model_revision: Option<String>,
        drafter_storage_fingerprint: Option<String>,
    },
}

/// Bounded source-selection reason safe for status and attribution.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Qwen3_5MtpSourceUnavailableReason {
    ConfiguredDrafterNotDiscovered,
    StandaloneArtifactInvalid,
    PairingIncompatible,
}
