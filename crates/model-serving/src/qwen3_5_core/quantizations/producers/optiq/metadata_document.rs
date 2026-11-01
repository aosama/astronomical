use std::collections::BTreeMap;

use serde::Deserialize;

/// The `method` value current mlx-optiq releases write for sensitivity-driven
/// mixed precision. Every other value routes to the static-allocation shape,
/// which is also what documents written before `method` was versioned carry.
const MIXED_PRECISION_METHOD: &str = "optiq_mixed_precision";

/// One per-module override as OptiQ restates it: bits and group size only,
/// with no `mode` key (affine is implied by the producer).
#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct OptiQPerLayerOverride {
    pub(super) bits: u32,
    pub(super) group_size: u32,
}

/// Document written by the sensitivity-driven `optiq_mixed_precision` method.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
#[allow(dead_code)]
pub(super) struct MixedPrecisionMetadataDocument {
    method: String,
    base_model: String,
    default_bits: u32,
    group_size: u32,
    bit_widths_used: Vec<u32>,
    // Keyed by the bit width rendered as text, as JSON object keys must be.
    layers_per_bit_width: BTreeMap<String, u64>,
    n_quantized_layers: u64,
    // Variants that reuse a sibling's allocation record where it came from
    // instead of re-measuring; both fields are provenance, never constraints.
    #[serde(default)]
    recipe_reused_from: Option<String>,
    #[serde(default)]
    recipe_note: Option<String>,
    #[serde(default)]
    sensitivity_measured_on: Option<String>,
    per_layer: BTreeMap<String, OptiQPerLayerOverride>,
}

/// Document written by the static (structure-rule) allocator, and by the
/// original sensitivity pipeline before `method` carried a version.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
#[allow(dead_code)]
pub(super) struct StaticMetadataDocument {
    method: String,
    base_model: String,
    reference: String,
    #[serde(default)]
    sensitivity_measured_on: Option<String>,
    target_bpw: f64,
    achieved_bpw: f64,
    n_high_bits: usize,
    n_low_bits: usize,
    threshold: f64,
    per_layer: BTreeMap<String, OptiQPerLayerOverride>,
}

/// The measured-bit-map documents the strict reader understands, one variant
/// per producer method so each keeps its own unknown-field rejection.
#[derive(Debug)]
pub(super) enum OptiQMetadataDocument {
    Mixed(MixedPrecisionMetadataDocument),
    Static(StaticMetadataDocument),
}

#[derive(Deserialize)]
struct MethodProbe {
    method: String,
}

/// Decides whether a `method` value names the sensitivity-driven shape.
pub(super) fn is_mixed_precision_method(method: &str) -> bool {
    method == MIXED_PRECISION_METHOD
}

impl OptiQMetadataDocument {
    /// Decodes the document by dispatching on `method`; a missing `method`
    /// or any shape violation is a strict decode error.
    pub(super) fn from_json_bytes(metadata_bytes: &[u8]) -> Result<Self, serde_json::Error> {
        let method_probe: MethodProbe = serde_json::from_slice(metadata_bytes)?;
        if is_mixed_precision_method(&method_probe.method) {
            serde_json::from_slice(metadata_bytes).map(Self::Mixed)
        } else {
            serde_json::from_slice(metadata_bytes).map(Self::Static)
        }
    }

    /// Consumes the document, keeping only the per-module bit map that the
    /// config cross-check needs.
    pub(super) fn into_per_layer(self) -> BTreeMap<String, OptiQPerLayerOverride> {
        match self {
            Self::Mixed(document) => document.per_layer,
            Self::Static(document) => document.per_layer,
        }
    }
}

#[cfg(test)]
mod metadata_document_tests;
