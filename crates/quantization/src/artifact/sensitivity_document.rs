//! `optiq/sensitivity.json`: the measured sweep itself, shipped with the quant.
//!
//! The metadata records the final bit assignment, which reproduces this
//! quant and nothing else. Re-aiming the same base at a different target
//! needs the underlying KL scores; without them a 6-bit build of an
//! already-measured 9B meant repeating a ten-hour sweep. The file is about
//! 50 KB. Writing it is never fatal: a conversion that succeeded must not
//! fail at the last step over a convenience file.
//!
//! The same document is the input to the CLI's `quantize plan` sub-verb,
//! which re-runs only the allocator, and to the acceptance parity journey,
//! which allocates from a Python-produced sweep and compares bit maps.

use std::path::Path;

use serde::{Deserialize, Serialize};

use std::collections::BTreeMap;

use crate::bit_width::{BitWidth, GroupSize};
use crate::error::QuantizationError;
use crate::sensitivity::{SensitivityMethod, SensitivityResult, SensitivityTable};

use super::json_file::{read_json, write_pretty_json};
use super::layout::{SENSITIVITY_FILE_NAME, artifact_path};

/// The only metric OptiQ records: KL divergence of the probe's logits from
/// the reference's.
const KL_DIVERGENCE_METRIC: &str = "kl_divergence_vs_reference";

#[cfg(test)]
#[path = "sensitivity_document_tests.rs"]
mod sensitivity_document_tests;

/// One layer entry in OptiQ's shape (string-keyed sensitivities).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SensitivityLayerEntry {
    pub layer_name: String,
    pub sensitivities: BTreeMap<String, f64>,
    /// Serialized as OptiQ's `param_count`.
    #[serde(rename = "param_count")]
    pub logical_parameter_count: u64,
}

/// The document.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OptiQSensitivityDocument {
    pub base_model: String,
    pub candidate_bits: Vec<u32>,
    pub group_size: u32,
    /// Always `kl_divergence_vs_reference`.
    pub metric: String,
    pub layers: Vec<SensitivityLayerEntry>,
}

impl OptiQSensitivityDocument {
    #[must_use]
    pub fn from_table(
        table: &SensitivityTable,
        base_model: &str,
        candidate_bits: &[u32],
        group_size: GroupSize,
    ) -> Self {
        let mut layers = Vec::with_capacity(table.results.len());
        for result in &table.results {
            layers.push(Self::layer_entry(result));
        }
        Self {
            base_model: base_model.to_owned(),
            candidate_bits: candidate_bits.to_vec(),
            group_size: group_size.elements(),
            metric: KL_DIVERGENCE_METRIC.to_owned(),
            layers,
        }
    }

    /// Back to a table for the allocator. Every key must parse as a valid
    /// bit width. The document does not record which reference the sweep
    /// measured against, so the table is marked as a bf16-reference sweep;
    /// the method only names the metadata strings, which a re-plan takes
    /// from its own request.
    pub fn to_table(&self) -> Result<SensitivityTable, QuantizationError> {
        let mut results = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            let mut kl_divergence_by_bit_width = BTreeMap::new();
            for (bits_key, kl_divergence) in &layer.sensitivities {
                let bits = parse_bits_key(bits_key)?;
                kl_divergence_by_bit_width.insert(BitWidth::new(bits)?, *kl_divergence);
            }
            results.push(SensitivityResult {
                module_name: layer.layer_name.clone(),
                kl_divergence_by_bit_width,
                logical_parameter_count: layer.logical_parameter_count,
            });
        }
        Ok(SensitivityTable {
            method: SensitivityMethod::ExactBfloat16Reference,
            results,
        })
    }

    pub fn write(&self, model_directory: &Path) -> Result<(), QuantizationError> {
        write_pretty_json(self, &artifact_path(model_directory, SENSITIVITY_FILE_NAME))
    }

    pub fn read(path: &Path) -> Result<Self, QuantizationError> {
        match read_json(path) {
            Ok(document) => Ok(document),
            Err(description) => Err(QuantizationError::SensitivityDocumentUnreadable {
                document_path: path.to_path_buf(),
                description,
            }),
        }
    }

    #[must_use]
    pub fn layer_entry(result: &SensitivityResult) -> SensitivityLayerEntry {
        let mut sensitivities = BTreeMap::new();
        for (bit_width, kl_divergence) in &result.kl_divergence_by_bit_width {
            sensitivities.insert(bit_width.bits().to_string(), *kl_divergence);
        }
        SensitivityLayerEntry {
            layer_name: result.module_name.clone(),
            sensitivities,
            logical_parameter_count: result.logical_parameter_count,
        }
    }
}

/// A sensitivity key is the bit width as a decimal string. A key that is
/// not a number reads as width 0, which no ladder accepts, so the error
/// names an unsupported width rather than passing a corrupt document on.
fn parse_bits_key(bits_key: &str) -> Result<u32, QuantizationError> {
    match bits_key.parse() {
        Ok(bits) => Ok(bits),
        Err(_) => Err(QuantizationError::UnsupportedBitWidth { bits: 0 }),
    }
}
