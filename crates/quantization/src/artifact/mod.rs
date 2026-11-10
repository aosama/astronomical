//! The files that turn a directory of requantized shards into a loadable,
//! self-describing artifact.
//!
//! The output format is OptiQ's, so three independent readers accept it:
//! stock mlx-lm (through `config.json`'s `quantization` document), OptiQ's
//! own tools (through `optiq/metadata.json` and `optiq/sensitivity.json`),
//! and Astronomical's model-serving crate, whose `MlxQuantizationDocument`
//! and `OptiQMetadata` parse these documents strictly and validate them
//! against each other at load. The integration round-trip test in model-serving
//! feeds this module's output straight into those readers.
//!
//! One rule from OptiQ's `artifacts.py`: anything OptiQ-specific goes in the
//! `optiq/` subfolder, and the repository root is left for files the wider
//! ecosystem reads (`config.json`, `generation_config.json`, the chat
//! template, the tokenizer, the weights).

pub mod converted_config;
pub mod generation_config;
pub mod json_file;
pub mod layout;
pub mod metadata;
pub mod model_card;
pub mod mtp_config;
pub mod multimodal_config;
pub mod quantization_config;
pub mod sensitivity_document;

pub use layout::{
    LEGACY_METADATA_FILE_NAME, METADATA_FILE_NAME, OPTIQ_SUBDIRECTORY, SENSITIVITY_FILE_NAME,
    artifact_path, resolve_metadata_path,
};
pub use metadata::{
    MixedPrecisionMetadata, OptiQArtifactMetadata, PerLayerEntry, StaticAllocationMetadata,
};
pub use quantization_config::QuantizationConfigDocument;
pub use sensitivity_document::OptiQSensitivityDocument;
