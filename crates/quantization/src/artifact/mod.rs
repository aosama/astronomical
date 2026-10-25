//! The files that turn a directory of requantized shards into a loadable,
//! self-describing artifact.
//!
//! The output format is OptiQ's, so three independent readers accept it:
//! stock mlx-lm (through `config.json`'s `quantization` document), OptiQ's
//! own tools (through `optiq/metadata.json` and `optiq/sensitivity.json`),
//! and Astronomical's model-serving crate, whose `OptiQQuantizationConfig`
//! and `OptiQMetadata` parse these documents strictly and validate them
//! against each other at load. The integration round-trip test in model-serving
//! feeds this module's output straight into those readers.
//!
//! One rule from OptiQ's `artifacts.py`: anything OptiQ-specific goes in the
//! `optiq/` subfolder, and the repository root is left for files the wider
//! ecosystem reads (`config.json`, `generation_config.json`, the chat
//! template, the tokenizer, the weights).

pub mod metadata;
pub mod quantization_config;
pub mod sensitivity_document;

use std::path::{Path, PathBuf};

#[cfg(test)]
#[path = "artifact_layout_tests.rs"]
mod artifact_layout_tests;

pub use metadata::OptiQArtifactMetadata;
pub use quantization_config::QuantizationConfigDocument;
pub use sensitivity_document::OptiQSensitivityDocument;

/// The subfolder holding everything OptiQ-specific.
pub const OPTIQ_SUBDIRECTORY: &str = "optiq";
pub const METADATA_FILE_NAME: &str = "metadata.json";
pub const SENSITIVITY_FILE_NAME: &str = "sensitivity.json";
/// The pre-0.4.18 root-level metadata name that published quants still carry
/// and that readers must keep accepting.
pub const LEGACY_METADATA_FILE_NAME: &str = "optiq_metadata.json";

/// Where `name` is written for an artifact at `model_directory`.
#[must_use]
pub fn artifact_path(model_directory: &Path, file_name: &str) -> PathBuf {
    model_directory.join(OPTIQ_SUBDIRECTORY).join(file_name)
}

/// An existing path for the metadata document, canonical location first.
#[must_use]
pub fn resolve_metadata_path(model_directory: &Path) -> Option<PathBuf> {
    let canonical = artifact_path(model_directory, METADATA_FILE_NAME);
    if canonical.is_file() {
        return Some(canonical);
    }
    let legacy = model_directory.join(LEGACY_METADATA_FILE_NAME);
    legacy.is_file().then_some(legacy)
}
