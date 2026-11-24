//! Where each of a layer's expert tensors sits inside that layer's records.
//!
//! Slots follow one canonical order — gate, up, down, each as weight, scales,
//! biases, then any other expert parameter by name — so every layer of a
//! uniform checkpoint lays out identically and a record reads in the order
//! the expert math consumes it. The stride rounds the record up to the
//! alignment so every record starts on a page.

use crate::atlas_manifest::{ATLAS_RECORD_ALIGNMENT_BYTES, AtlasExpertLayer, AtlasRecordSlot};
use crate::source_checkpoint::SourceExpertLayer;

const PROJECTION_ORDER: [&str; 3] = ["gate_proj", "up_proj", "down_proj"];
const PARAMETER_ORDER: [&str; 3] = ["weight", "scales", "biases"];

/// The layer file name for `layer_index`, zero-padded so files sort by layer.
#[must_use]
pub fn expert_layer_file_name(layer_index: usize) -> String {
    format!(
        "{}/layer-{layer_index:03}.atlas",
        crate::atlas_manifest::EXPERT_LAYER_DIRECTORY_NAME
    )
}

/// Plans one layer's records from its source tensors.
#[must_use]
pub fn plan_layer_records(
    source_expert_layer: &SourceExpertLayer,
    expert_count: usize,
) -> AtlasExpertLayer {
    let mut ordered_tensors: Vec<_> = source_expert_layer.expert_tensors.iter().collect();
    ordered_tensors.sort_by_key(|source_tensor| canonical_slot_order(&source_tensor.tensor_name));
    let mut offset_in_record = 0;
    let mut slots = Vec::with_capacity(ordered_tensors.len());
    for source_tensor in ordered_tensors {
        let bytes_per_expert = source_tensor.payload_byte_count / expert_count as u64;
        slots.push(AtlasRecordSlot {
            tensor_name: source_tensor.tensor_name.clone(),
            dtype: source_tensor.dtype.clone(),
            shape: source_tensor.shape.clone(),
            offset_in_record,
            bytes_per_expert,
        });
        offset_in_record += bytes_per_expert;
    }
    AtlasExpertLayer {
        layer_index: source_expert_layer.layer_index,
        file_name: expert_layer_file_name(source_expert_layer.layer_index),
        expert_count,
        record_byte_count: offset_in_record,
        record_stride_bytes: offset_in_record.div_ceil(ATLAS_RECORD_ALIGNMENT_BYTES)
            * ATLAS_RECORD_ALIGNMENT_BYTES,
        slots,
    }
}

/// `(projection rank, parameter rank, name)`: known projections and
/// parameters first in serving order, anything else after them by name.
fn canonical_slot_order(tensor_name: &str) -> (usize, usize, String) {
    let mut name_parts = tensor_name.rsplit('.');
    let parameter_name = name_parts.next().unwrap_or_default();
    let projection_name = name_parts.next().unwrap_or_default();
    let rank_of = |known_names: &[&str], name: &str| {
        known_names
            .iter()
            .position(|known_name| *known_name == name)
            .unwrap_or(known_names.len())
    };
    (
        rank_of(&PROJECTION_ORDER, projection_name),
        rank_of(&PARAMETER_ORDER, parameter_name),
        tensor_name.to_owned(),
    )
}

#[cfg(test)]
#[path = "record_layout_tests.rs"]
mod record_layout_tests;
