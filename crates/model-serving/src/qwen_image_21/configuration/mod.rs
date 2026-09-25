//! Typed configuration profiles for the reviewed Qwen-Image-2.1 MLX artifact.
//!
//! Each component config is parsed strictly (`deny_unknown_fields`) and pinned to the reviewed
//! geometry so an engine only ever runs against the package shape the hermetic oracles were
//! generated from. One file per component — `pipeline`, `transformer`, `text_encoder`, `vae`,
//! `scheduler` — and this module owns what they share: the typed rejection, the strict document
//! parsing, and the reviewed 4-bit affine quantization contract.

mod pipeline;
mod scheduler;
mod text_encoder;
mod transformer;
mod vae;

pub use pipeline::QwenImage21PipelineConfig;
pub use scheduler::QwenImage21SchedulerConfig;
pub use text_encoder::QwenImage21TextEncoderConfig;
pub use transformer::QwenImage21TransformerConfig;
pub use vae::QwenImage21VaeConfig;

use serde::Deserialize;
use thiserror::Error;

use crate::strict_json::DuplicateAwareJsonValue;

/// A rejected Qwen-Image-2.1 configuration before any weight descriptor reaches an engine.
#[derive(Debug, Error)]
pub enum QwenImage21ConfigError {
    #[error("malformed Qwen-Image-2.1 {document} configuration")]
    Malformed {
        document: &'static str,
        #[source]
        source: serde_json::Error,
    },
    #[error("unsupported Qwen-Image-2.1 profile in {document}: {field}")]
    UnsupportedProfile {
        document: &'static str,
        field: &'static str,
    },
}

/// Parses one config document with duplicate-key detection before the typed read.
///
/// `document` is the artifact-relative path, so the rejection names the file a maintainer has
/// to look at without exposing a local directory.
pub(super) fn parse_document<T: serde::de::DeserializeOwned>(
    json_bytes: &[u8],
    document: &'static str,
) -> Result<T, QwenImage21ConfigError> {
    let duplicate_aware = serde_json::from_slice::<DuplicateAwareJsonValue>(json_bytes)
        .map_err(|source| QwenImage21ConfigError::Malformed { document, source })?;
    serde_json::from_value(duplicate_aware.0)
        .map_err(|source| QwenImage21ConfigError::Malformed { document, source })
}

/// Fails with `UnsupportedProfile` unless the reviewed value holds.
pub(super) fn require(
    condition: bool,
    document: &'static str,
    field: &'static str,
) -> Result<(), QwenImage21ConfigError> {
    condition
        .then_some(())
        .ok_or(QwenImage21ConfigError::UnsupportedProfile { document, field })
}

/// The quantization block every reviewed component config carries.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct QuantizationDocument {
    bits: u32,
    group_size: u32,
    mode: String,
}

/// The reviewed artifact quantizes every projection the same way: 4-bit affine in 64-wide
/// groups. Anything else means the package is not the one the profiles describe.
pub(super) fn reviewed_quantization(
    quantization: &QuantizationDocument,
    document: &'static str,
) -> Result<(), QwenImage21ConfigError> {
    require(
        quantization.bits == 4 && quantization.group_size == 64 && quantization.mode == "affine",
        document,
        "quantization",
    )
}

/// Review bounds: 4-bit affine dual-quantized linears store `U32` weight rows of
/// `in_features * bits / 32` packed words plus `BF16` per-group scales and biases of
/// `in_features / group_size`.
#[must_use]
pub const fn quantized_row_count(input_features: usize, bits: u32) -> usize {
    input_features * (bits as usize) / 32
}

#[must_use]
pub const fn quantized_group_count(input_features: usize, group_size: u32) -> usize {
    input_features / (group_size as usize)
}
