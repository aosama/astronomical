//! Validated multi-token-prediction draft depth owned beside memory admission.
//!
//! MTP memory admission projects per-depth byte growth, so the depth type it
//! selects among must live in the same package and stay free of family imports.

use thiserror::Error;

/// Validated fixed MTP proposal depth supported by Astronomical.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct MtpDraftDepth(u8);

impl MtpDraftDepth {
    pub const MINIMUM: u8 = 1;
    pub const MAXIMUM: u8 = 3;
    pub const DEPTH_ONE: Self = Self(1);

    /// Validates a configured or artifact-declared draft depth.
    pub fn new(depth: u8) -> Result<Self, MtpDraftDepthError> {
        if (Self::MINIMUM..=Self::MAXIMUM).contains(&depth) {
            Ok(Self(depth))
        } else {
            Err(MtpDraftDepthError { depth })
        }
    }

    #[must_use]
    pub const fn get(self) -> u8 {
        self.0
    }
}

/// A draft depth outside Astronomical's fixed supported range.
#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
#[error("MTP draft depth must be between 1 and 3")]
pub struct MtpDraftDepthError {
    depth: u8,
}
