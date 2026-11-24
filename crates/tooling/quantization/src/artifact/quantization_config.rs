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

use super::json_file::{read_json, write_pretty_json};

#[cfg(test)]
#[path = "quantization_config_tests.rs"]
mod quantization_config_tests;

/// One module's written parameters.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ModuleQuantizationEntry {
    pub group_size: u32,
    pub bits: u32,
    /// Absent in documents mlx-lm's convert writes: a module entry inherits
    /// the document's top-level mode. Present where a producer restates it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode: Option<String>,
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
        let mut modules = BTreeMap::new();
        for (module_name, (bits, group_size)) in written_quantization_by_module {
            modules.insert(
                module_name.clone(),
                ModuleQuantizationEntry {
                    group_size: *group_size,
                    bits: *bits,
                    mode: None,
                },
            );
        }
        Self {
            group_size: default_group_size,
            bits: default_bits,
            mode: AFFINE_MODE.to_owned(),
            modules,
        }
    }

    /// Reads the source `config.json`, replaces (or inserts) both keys with
    /// this document, and writes the result to the output `config.json`,
    /// preserving every other key verbatim.
    pub fn write_into_config(
        &self,
        source_config_path: &Path,
        output_config_path: &Path,
    ) -> Result<(), QuantizationError> {
        let mut config_entries: serde_json::Map<String, serde_json::Value> =
            match read_json(source_config_path) {
                Ok(config_entries) => config_entries,
                Err(description) => return Err(config_unreadable(source_config_path, description)),
            };
        let document_value = match serde_json::to_value(self) {
            Ok(document_value) => document_value,
            Err(serialize_error) => {
                return Err(config_unreadable(
                    source_config_path,
                    serialize_error.to_string(),
                ));
            }
        };
        for document_key in QUANTIZATION_DOCUMENT_KEYS {
            config_entries.insert(document_key.to_owned(), document_value.clone());
        }
        write_pretty_json(&config_entries, output_config_path)
    }

    /// Reads the quantization document of an existing artifact (the source
    /// of a requantization), per-module overrides included.
    pub fn read_from_config(config_path: &Path) -> Result<Option<Self>, QuantizationError> {
        let config_value: serde_json::Value = match read_json(config_path) {
            Ok(config_value) => config_value,
            Err(description) => return Err(config_unreadable(config_path, description)),
        };
        for document_key in QUANTIZATION_DOCUMENT_KEYS {
            let document_value = &config_value[document_key];
            if document_value.is_object() {
                return match serde_json::from_value(document_value.clone()) {
                    Ok(document) => Ok(Some(document)),
                    Err(parse_error) => {
                        Err(config_unreadable(config_path, parse_error.to_string()))
                    }
                };
            }
        }
        Ok(None)
    }

    /// The parameters a source tensor was written with: its module entry
    /// when present, else the global defaults.
    #[must_use]
    pub fn source_parameters_for(&self, module_name: &str) -> (u32, u32) {
        match self.modules.get(module_name) {
            Some(module_entry) => (module_entry.bits, module_entry.group_size),
            None => (self.bits, self.group_size),
        }
    }
}

/// The only quantization mode this crate writes.
const AFFINE_MODE: &str = "affine";

/// mlx-lm reads `quantization`; other tooling reads `quantization_config`.
/// Both carry the same document, and `quantization` wins on read.
const QUANTIZATION_DOCUMENT_KEYS: [&str; 2] = ["quantization", "quantization_config"];

fn config_unreadable(config_path: &Path, description: String) -> QuantizationError {
    QuantizationError::ModelConfigUnreadable {
        config_path: config_path.to_path_buf(),
        description,
    }
}
