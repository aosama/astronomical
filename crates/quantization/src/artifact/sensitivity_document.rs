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

use crate::bit_width::GroupSize;
use crate::error::QuantizationError;
use crate::sensitivity::{SensitivityResult, SensitivityTable};

#[cfg(test)]
#[path = "sensitivity_document_tests.rs"]
mod sensitivity_document_tests;

/// One layer entry in OptiQ's shape (string-keyed sensitivities).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SensitivityLayerEntry {
    pub layer_name: String,
    pub sensitivities: std::collections::BTreeMap<String, f64>,
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
        let _ = (table, base_model, candidate_bits, group_size);
        todo!("translate each SensitivityResult to a layer entry with string bit keys")
    }

    /// Back to a table for the allocator. Every key must parse as a valid
    /// bit width.
    pub fn to_table(&self) -> Result<SensitivityTable, QuantizationError> {
        todo!(
            "translate layer entries back; the method is recorded as unknown-exact since the document does not carry it"
        )
    }

    pub fn write(&self, model_directory: &Path) -> Result<(), QuantizationError> {
        let _ = model_directory;
        todo!("create optiq/ and write pretty JSON")
    }

    pub fn read(path: &Path) -> Result<Self, QuantizationError> {
        let _ = path;
        todo!("read and parse; map I/O and JSON failures to SensitivityDocumentUnreadable")
    }

    #[must_use]
    pub fn layer_entry(result: &SensitivityResult) -> SensitivityLayerEntry {
        let _ = result;
        todo!("string-key the bit widths")
    }
}
