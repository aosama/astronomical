//! `optiq/metadata.json`: the provenance and the final per-module bit map.
//!
//! Published OptiQ artifacts carry one of two layouts, chosen by `method`:
//! the sensitivity-driven `optiq_mixed_precision` layout (default bits, the
//! widths used and how many modules sit at each) and the static allocator's
//! layout (reference, target and achieved bits per weight). Astronomical's
//! strict `OptiQMetadata` reader accepts exactly these two shapes, rejects
//! any other key, and requires every `per_layer` entry to match the
//! `config.json` override for the same module. So `per_layer` lists only the
//! modules that were quantized, with the parameters they were written at,
//! never the plan: the requantizer can drop a module to group size 32 or
//! pass a router through.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::allocation::AllocationResult;
use crate::error::QuantizationError;
use crate::sensitivity::SensitivityMethod;

use super::json_file::write_pretty_json;
use super::layout::{METADATA_FILE_NAME, artifact_path};

#[cfg(test)]
#[path = "metadata_tests.rs"]
mod metadata_tests;

/// One quantized module's written parameters. OptiQ restates bits and group
/// size only; affine is implied by the producer.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PerLayerEntry {
    pub bits: u32,
    pub group_size: u32,
}

/// The `optiq_mixed_precision` layout, written by sensitivity-driven
/// conversions.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MixedPrecisionMetadata {
    pub method: String,
    pub base_model: String,
    /// The lowest written width, the conversion's baseline.
    pub default_bits: u32,
    pub group_size: u32,
    pub bit_widths_used: Vec<u32>,
    /// Keyed by the width as text, as JSON object keys must be.
    pub layers_per_bit_width: BTreeMap<String, u64>,
    pub n_quantized_layers: u64,
    /// Provenance for variants that reuse a sibling's allocation; never
    /// written by this crate, kept so published documents read back.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recipe_reused_from: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recipe_note: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sensitivity_measured_on: Option<String>,
    pub per_layer: BTreeMap<String, PerLayerEntry>,
}

/// The static allocator's layout, also written by OptiQ's measured pipeline
/// before `method` carried a version.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StaticAllocationMetadata {
    pub method: String,
    pub base_model: String,
    /// `bf16`, `uniform_4bit`, or `structural_rules`.
    pub reference: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sensitivity_measured_on: Option<String>,
    pub target_bpw: f64,
    pub achieved_bpw: f64,
    /// Modules at the ladder's highest width.
    pub n_high_bits: usize,
    /// Modules at the ladder's lowest width.
    pub n_low_bits: usize,
    /// Kept at 0.0 for format compatibility with pre-multi-tier readers.
    pub threshold: f64,
    pub per_layer: BTreeMap<String, PerLayerEntry>,
}

/// The document, in whichever layout its method writes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum OptiQArtifactMetadata {
    MixedPrecision(MixedPrecisionMetadata),
    StaticAllocation(StaticAllocationMetadata),
}

impl OptiQArtifactMetadata {
    /// An empty document in the layout `method` writes; `per_layer` is filled
    /// by [`Self::describe_written_artifact`] once the weights exist.
    #[must_use]
    pub fn for_method(method: SensitivityMethod, base_model: &str, group_size: u32) -> Self {
        let method_name = Self::method_name(method).to_owned();
        if method == SensitivityMethod::StaticStructuralRules {
            return Self::StaticAllocation(StaticAllocationMetadata {
                method: method_name,
                base_model: base_model.to_owned(),
                reference: Self::reference_name(method).to_owned(),
                sensitivity_measured_on: None,
                target_bpw: 0.0,
                achieved_bpw: 0.0,
                n_high_bits: 0,
                n_low_bits: 0,
                threshold: 0.0,
                per_layer: BTreeMap::new(),
            });
        }
        Self::MixedPrecision(MixedPrecisionMetadata {
            method: method_name,
            base_model: base_model.to_owned(),
            default_bits: 0,
            group_size,
            bit_widths_used: Vec::new(),
            layers_per_bit_width: BTreeMap::new(),
            n_quantized_layers: 0,
            recipe_reused_from: None,
            recipe_note: None,
            sensitivity_measured_on: None,
            per_layer: BTreeMap::new(),
        })
    }

    /// The planned document from an allocation: the allocation's figures and
    /// its quantized decisions (keep-bfloat16 modules are not listed).
    #[must_use]
    pub fn from_allocation(
        allocation: &AllocationResult,
        base_model: &str,
        method: SensitivityMethod,
    ) -> Self {
        let mut planned_quantization_by_module = BTreeMap::new();
        let mut group_size = DEFAULT_GROUP_SIZE;
        for decision in &allocation.decisions {
            if !decision.bit_width.keeps_bfloat16() {
                group_size = decision.group_size.elements();
                let planned_parameters = (decision.bit_width.bits(), group_size);
                planned_quantization_by_module
                    .insert(decision.module_name.clone(), planned_parameters);
            }
        }
        let mut metadata = Self::for_method(method, base_model, group_size);
        if let Self::StaticAllocation(static_metadata) = &mut metadata {
            static_metadata.target_bpw = allocation.target_bits_per_weight;
            static_metadata.achieved_bpw = allocation.achieved_bits_per_weight;
            static_metadata.n_high_bits = allocation.module_count_at_highest_width;
            static_metadata.n_low_bits = allocation.module_count_at_lowest_width;
        }
        metadata.describe_written_artifact(&planned_quantization_by_module);
        metadata
    }

    /// Makes `per_layer` describe exactly the weights that were written,
    /// `(bits, group_size)` per quantized module, and recomputes the mixed
    /// layout's summary from it. A planned module that was not written
    /// (passed through at bfloat16) leaves the map.
    pub fn describe_written_artifact(
        &mut self,
        written_quantization_by_module: &BTreeMap<String, (u32, u32)>,
    ) {
        let mut per_layer = BTreeMap::new();
        for (module_name, (bits, group_size)) in written_quantization_by_module {
            per_layer.insert(
                module_name.clone(),
                PerLayerEntry {
                    bits: *bits,
                    group_size: *group_size,
                },
            );
        }
        match self {
            Self::MixedPrecision(mixed_metadata) => {
                summarize_mixed_precision(mixed_metadata, per_layer);
            }
            Self::StaticAllocation(static_metadata) => static_metadata.per_layer = per_layer,
        }
    }

    /// The per-module bit map, in either layout.
    #[must_use]
    pub fn per_layer(&self) -> &BTreeMap<String, PerLayerEntry> {
        match self {
            Self::MixedPrecision(mixed_metadata) => &mixed_metadata.per_layer,
            Self::StaticAllocation(static_metadata) => &static_metadata.per_layer,
        }
    }

    /// Writes to `<model_directory>/optiq/metadata.json`, creating the folder.
    pub fn write(&self, model_directory: &Path) -> Result<(), QuantizationError> {
        write_pretty_json(self, &artifact_path(model_directory, METADATA_FILE_NAME))
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

/// OptiQ's default group size, recorded when an allocation quantizes nothing.
const DEFAULT_GROUP_SIZE: u32 = 64;

/// Sets the per-layer map and the summary the mixed layout derives from it:
/// the lowest width as the default, the sorted widths used, and the count at
/// each.
fn summarize_mixed_precision(
    mixed_metadata: &mut MixedPrecisionMetadata,
    per_layer: BTreeMap<String, PerLayerEntry>,
) {
    let mut module_count_by_bits: BTreeMap<u32, u64> = BTreeMap::new();
    for per_layer_entry in per_layer.values() {
        *module_count_by_bits
            .entry(per_layer_entry.bits)
            .or_insert(0) += 1;
    }
    mixed_metadata.bit_widths_used = module_count_by_bits.keys().copied().collect();
    mixed_metadata.default_bits = mixed_metadata.bit_widths_used.first().copied().unwrap_or(0);
    mixed_metadata.layers_per_bit_width = BTreeMap::new();
    for (bits, module_count) in module_count_by_bits {
        mixed_metadata
            .layers_per_bit_width
            .insert(bits.to_string(), module_count);
    }
    mixed_metadata.n_quantized_layers = per_layer.len() as u64;
    mixed_metadata.per_layer = per_layer;
}
