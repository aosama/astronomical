//! The `quantization` and `quantization_config` documents in `config.json`.
//!
//! Both keys carry the same document: global defaults `group_size`, `bits`,
//! `mode: "affine"`, plus one `{group_size, bits, mode}` entry per module
//! that was quantized. Modules not listed were written at bfloat16. mlx-lm
//! reads `quantization`; some tooling reads `quantization_config`; Astronomical
//! validates both and requires them to agree.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::QuantizationError;

#[cfg(test)]
#[path = "quantization_config_tests.rs"]
mod quantization_config_tests;

/// One module's written parameters.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ModuleQuantizationEntry {
    pub group_size: u32,
    pub bits: u32,
    pub mode: String,
}

/// The document as serialized: defaults plus flattened per-module entries.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct QuantizationConfigDocument {
    pub group_size: u32,
    pub bits: u32,
    pub mode: String,
    #[serde(flatten)]
    pub modules: BTreeMap<String, ModuleQuantizationEntry>,
}

impl QuantizationConfigDocument {
    /// Builds the document from what the requantizer actually wrote.
    ///
    /// The defaults are OptiQ's `{"group_size": 64, "bits": 4, "mode":
    /// "affine"}` unless the caller overrides them; the per-module entries
    /// come from `written_quantization_by_module`.
    #[must_use]
    pub fn from_written_modules(
        default_group_size: u32,
        default_bits: u32,
        written_quantization_by_module: &BTreeMap<String, (u32, u32)>,
    ) -> Self {
        let _ = (
            default_group_size,
            default_bits,
            written_quantization_by_module,
        );
        todo!("map each (bits, group_size) to an affine entry under the module name")
    }

    /// Reads the source `config.json`, replaces (or inserts) both keys with
    /// this document, and writes the result to the output `config.json`,
    /// preserving every other key verbatim.
    pub fn write_into_config(
        &self,
        source_config_path: &Path,
        output_config_path: &Path,
    ) -> Result<(), QuantizationError> {
        let _ = (source_config_path, output_config_path);
        todo!(
            "read source JSON as a map, set quantization and quantization_config, write pretty JSON"
        )
    }

    /// Reads the quantization document of an existing artifact (the source
    /// of a requantization), per-module overrides included.
    pub fn read_from_config(config_path: &Path) -> Result<Option<Self>, QuantizationError> {
        let _ = config_path;
        todo!("parse config.json; return None when neither key is present")
    }

    /// The parameters a source tensor was written with: its module entry
    /// when present, else the global defaults.
    #[must_use]
    pub fn source_parameters_for(&self, module_name: &str) -> (u32, u32) {
        let _ = module_name;
        todo!("module entry bits and group_size, else the defaults")
    }
}
