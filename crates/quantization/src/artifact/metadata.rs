//! `optiq/metadata.json`: the provenance and the final per-module bit map.
//!
//! Astronomical's `OptiQMetadata` reader accepts exactly this shape and
//! requires every `per_layer` entry to match the `config.json` override for
//! the same module, so the document must describe the artifact that was
//! written, not the allocation that was planned (the requantizer can drop a
//! module to group size 32 or pass a router through).

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::allocation::AllocationResult;
use crate::error::QuantizationError;
use crate::sensitivity::SensitivityMethod;

#[cfg(test)]
#[path = "metadata_tests.rs"]
mod metadata_tests;

/// One module's written parameters; `group_size` is `None` for a module
/// left at bfloat16 (recorded with `bits: 16`).
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PerLayerEntry {
    pub bits: u32,
    pub group_size: Option<u32>,
}

/// The document.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OptiQArtifactMetadata {
    /// `optiq_mixed_precision` or `static_mixed_precision`.
    pub method: String,
    pub base_model: String,
    /// `bf16`, `uniform_4bit`, or `structural_rules`.
    pub reference: String,
    /// Serialized as OptiQ's `target_bpw`.
    #[serde(rename = "target_bpw")]
    pub target_bits_per_weight: f64,
    /// Serialized as OptiQ's `achieved_bpw`.
    #[serde(rename = "achieved_bpw")]
    pub achieved_bits_per_weight: f64,
    /// Modules at the ladder's highest width; serialized as `n_high_bits`.
    #[serde(rename = "n_high_bits")]
    pub module_count_at_highest_width: usize,
    /// Modules at the ladder's lowest width; serialized as `n_low_bits`.
    #[serde(rename = "n_low_bits")]
    pub module_count_at_lowest_width: usize,
    /// Kept at 0.0 for format compatibility with pre-multi-tier readers.
    pub threshold: f64,
    pub per_layer: BTreeMap<String, PerLayerEntry>,
    /// The producing tool, so a consumer can tell a Rust-built artifact from
    /// a Python-built one. Extra keys are accepted by OptiQ's reader and by
    /// Astronomical's provenance-only path.
    pub producer: String,
}

impl OptiQArtifactMetadata {
    /// Builds the planned document from an allocation.
    #[must_use]
    pub fn from_allocation(
        allocation: &AllocationResult,
        base_model: &str,
        method: SensitivityMethod,
    ) -> Self {
        let _ = (allocation, base_model, method);
        todo!(
            "map method and reference strings, copy the figures, and build per_layer from the decisions"
        )
    }

    /// Makes `per_layer` describe the weights that were written: a planned
    /// module with no written entry stayed bfloat16 (`bits: 16, group_size:
    /// null`); every written module takes its written parameters.
    pub fn describe_written_artifact(
        &mut self,
        written_quantization_by_module: &BTreeMap<String, (u32, u32)>,
    ) {
        let _ = written_quantization_by_module;
        todo!("reconcile per_layer against what the requantizer wrote")
    }

    /// Writes to `<model_directory>/optiq/metadata.json`, creating the folder.
    pub fn write(&self, model_directory: &Path) -> Result<(), QuantizationError> {
        let _ = model_directory;
        todo!("create optiq/ and write pretty JSON")
    }

    /// The method string OptiQ writes for a sensitivity method.
    #[must_use]
    pub fn method_name(method: SensitivityMethod) -> &'static str {
        match method {
            SensitivityMethod::ExactBfloat16Reference
            | SensitivityMethod::ExactUniform4BitReference => "optiq_mixed_precision",
            SensitivityMethod::StaticStructuralRules => "static_mixed_precision",
        }
    }

    /// The reference string OptiQ writes for a sensitivity method.
    #[must_use]
    pub fn reference_name(method: SensitivityMethod) -> &'static str {
        match method {
            SensitivityMethod::ExactBfloat16Reference => "bf16",
            SensitivityMethod::ExactUniform4BitReference => "uniform_4bit",
            SensitivityMethod::StaticStructuralRules => "structural_rules",
        }
    }
}
