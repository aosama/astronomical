//! Measured reply speed of a catalog entry: how many tokens a second it writes
//! for an essay and for code, after a short prompt and after a prompt of about
//! 10,000 tokens.
//!
//! The figures come from one measurement method (temperature 0.6, 500 tokens
//! out, the median of three interleaved rounds after a warm-up, through the
//! Development engine with the `astro_ssd_prompt_cache` off) on one named Mac,
//! so every figure carries where and with which Astronomical version it was
//! measured. Whole tokens are precise enough to tell builds apart, and integers
//! keep the entry comparable for equality.

use serde::Deserialize;

use crate::download_catalog::DownloadCatalogError;

// Above any rate a Mac reaches today by a wide margin; a larger figure is a
// typing mistake, not a measurement.
const MAXIMUM_TOKENS_PER_SECOND: u16 = 1_000;
const MAXIMUM_MEASURED_ON_BYTES: usize = 64;
const MAXIMUM_VERSION_BYTES: usize = 16;

/// Reply speed in whole tokens a second for each of the four measured tasks.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DownloadCatalogTaskSpeeds {
    pub essay_short_prompt: u16,
    pub essay_long_prompt: u16,
    pub code_short_prompt: u16,
    pub code_long_prompt: u16,
}

impl DownloadCatalogTaskSpeeds {
    const fn all(self) -> [u16; 4] {
        [
            self.essay_short_prompt,
            self.essay_long_prompt,
            self.code_short_prompt,
            self.code_long_prompt,
        ]
    }
}

/// The measured speeds of one entry and where they were measured.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DownloadCatalogMeasuredSpeed {
    measured_on: String,
    astronomical_version: String,
    tokens_per_second: DownloadCatalogTaskSpeeds,
}

impl DownloadCatalogMeasuredSpeed {
    /// The Mac the figures were measured on, in words a person reads.
    #[must_use]
    pub fn measured_on(&self) -> &str {
        &self.measured_on
    }

    #[must_use]
    pub fn astronomical_version(&self) -> &str {
        &self.astronomical_version
    }

    #[must_use]
    pub const fn tokens_per_second(&self) -> DownloadCatalogTaskSpeeds {
        self.tokens_per_second
    }
}

/// Every rate must be a real measurement (1 to 1,000), the Mac must be named,
/// and the version must look like one; anything else is refused rather than
/// shown as a figure people would trust.
pub(crate) fn validate_measured_speed(
    measured_speed: Option<&DownloadCatalogMeasuredSpeed>,
    entry_index: usize,
) -> Result<(), DownloadCatalogError> {
    let Some(measured_speed) = measured_speed else {
        return Ok(());
    };
    let rates_are_plausible = measured_speed
        .tokens_per_second
        .all()
        .iter()
        .all(|tokens_per_second| (1..=MAXIMUM_TOKENS_PER_SECOND).contains(tokens_per_second));
    let measured_on_is_named =
        is_bounded_label(&measured_speed.measured_on, MAXIMUM_MEASURED_ON_BYTES);
    let version_is_numeric =
        is_bounded_label(&measured_speed.astronomical_version, MAXIMUM_VERSION_BYTES)
            && measured_speed
                .astronomical_version
                .split('.')
                .all(|version_part| {
                    !version_part.is_empty()
                        && version_part.bytes().all(|byte| byte.is_ascii_digit())
                });
    if rates_are_plausible && measured_on_is_named && version_is_numeric {
        Ok(())
    } else {
        Err(DownloadCatalogError::InvalidMeasuredSpeed { entry_index })
    }
}

fn is_bounded_label(label: &str, maximum_bytes: usize) -> bool {
    !label.trim().is_empty() && label.len() <= maximum_bytes && !label.chars().any(char::is_control)
}

#[cfg(test)]
#[path = "catalog_speed_tests.rs"]
mod catalog_speed_tests;
