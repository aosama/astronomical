use std::collections::BTreeMap;

use serde::Deserialize;

/// One per-module entry as OptiQ restates it: bits and group size only,
/// with no `mode` key (affine is implied by the producer). The group size is
/// absent (`null`) for a module OptiQ's streaming convert left at bfloat16.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct OptiQPerLayerOverride {
    pub(super) bits: u32,
    pub(super) group_size: Option<u32>,
}

/// The summary layout: default bits, the widths used and how many modules
/// sit at each. Carried by published `optiq_mixed_precision` artifacts that
/// predate the pipeline layout or were assembled from a sibling's recipe.
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

/// The pipeline layout: the reference, the target and achieved bits per
/// weight and the counts at the ladder's ends. mlx-optiq's conversion
/// pipeline writes it for every method, the measured `optiq_mixed_precision`
/// as well as the structural `static_mixed_precision`.
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
/// per layout so each keeps its own unknown-field rejection.
#[derive(Debug)]
pub(super) enum OptiQMetadataDocument {
    Mixed(MixedPrecisionMetadataDocument),
    Static(StaticMetadataDocument),
}

/// The two fields that decide the layout. `method` must be present in
/// both. `default_bits` exists only in the summary layout, so its presence
/// selects it: the method name cannot, because `optiq_mixed_precision`
/// labels documents in both layouts.
#[derive(Deserialize)]
struct LayoutProbe {
    #[allow(dead_code)]
    method: String,
    #[serde(default)]
    default_bits: Option<serde_json::Value>,
}

impl OptiQMetadataDocument {
    /// Decodes the document by dispatching on its layout; a missing `method`
    /// or any shape violation is a strict decode error.
    pub(super) fn from_json_bytes(metadata_bytes: &[u8]) -> Result<Self, serde_json::Error> {
        let layout_probe: LayoutProbe = serde_json::from_slice(metadata_bytes)?;
        if layout_probe.default_bits.is_some() {
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
