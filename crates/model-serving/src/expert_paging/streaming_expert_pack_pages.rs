//! Routed page manifests over per-expert pack files plus GPU-side assembly.
//!
//! The virtual layout matches the shard path exactly: tensor-major, with the
//! normalized expert order occupying consecutive page slots. Per-file tensor
//! names carry an `@{slot}` suffix because the generic bounded reader merges
//! loaded tensors by name; the assembly step then rebuilds the canonical
//! `[expert_count, ...]` arrays from slot-ordered pieces.

#[cfg(feature = "direct-mlx")]
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::quantized_expert_manifest::{
    QuantizedExpertPageManifest, QuantizedExpertShardManifest, QuantizedExpertSourceInterval,
    QuantizedExpertTensorRange, QuantizedTensorSource, build_page_slot_by_global_expert_id,
};
use super::streaming_expert_packs::{
    PACK_HEADER_BYTES, PACK_SEGMENT_ALIGNMENT_BYTES, StreamingExpertPackError, align_up,
    validate_routed_expert_ids,
};
use crate::expert_paging::QuantizedExpertLayerPlan;

/// Builds a routed page manifest whose intervals point at per-expert pack
/// files. The virtual layout matches the shard path exactly: tensor-major,
/// with the normalized expert order occupying consecutive page slots.
pub fn build_streaming_expert_page_manifest(
    layer_plan: &QuantizedExpertLayerPlan,
    expert_file_paths: &[PathBuf],
    expert_ids: &[usize],
) -> Result<QuantizedExpertPageManifest, StreamingExpertPackError> {
    let normalized_expert_ids = validate_routed_expert_ids(expert_ids, layer_plan.expert_capacity)?;
    let mut source_manifests = Vec::with_capacity(normalized_expert_ids.len());
    for (page_slot, &expert_id) in normalized_expert_ids.iter().enumerate() {
        source_manifests.push(build_expert_file_shard_manifest(
            layer_plan,
            &expert_file_paths[expert_id],
            page_slot,
        )?);
    }
    let payload_byte_count = source_manifests
        .iter()
        .map(|manifest| manifest.payload_byte_count)
        .sum();
    let page_slot_by_global_expert_id =
        build_page_slot_by_global_expert_id(&normalized_expert_ids, layer_plan.expert_capacity)
            .map_err(|error| StreamingExpertPackError::ExpertIdValidation {
                description: error.to_string(),
            })?;
    Ok(QuantizedExpertPageManifest {
        expert_ids: normalized_expert_ids,
        page_slot_by_global_expert_id,
        source_manifests,
        payload_byte_count,
    })
}

fn build_expert_file_shard_manifest(
    layer_plan: &QuantizedExpertLayerPlan,
    expert_file_path: &Path,
    page_slot: usize,
) -> Result<QuantizedExpertShardManifest, StreamingExpertPackError> {
    let mut tensor_ranges = Vec::with_capacity(layer_plan.tensor_sources.len());
    let mut source_intervals = Vec::with_capacity(layer_plan.tensor_sources.len());
    let mut next_virtual_payload_offset = 0_u64;
    let mut next_segment_offset_bytes = PACK_HEADER_BYTES;
    for tensor_source in &layer_plan.tensor_sources {
        let tensor_byte_count = tensor_source.bytes_per_expert;
        let mangled_tensor_name = format!(
            "{page_tensor_name}@{page_slot}",
            page_tensor_name = expert_page_tensor_name(
                &tensor_source.projection_name,
                &tensor_source.parameter_name,
            )
        );
        tensor_ranges.push(QuantizedExpertTensorRange {
            tensor_name: mangled_tensor_name.clone(),
            projection_name: tensor_source.projection_name.clone(),
            parameter_name: tensor_source.parameter_name.clone(),
            dtype: tensor_source.dtype,
            // The leading page-slot axis of one: concatenating the slot-ordered
            // pieces rebuilds the shard path's `[expert_count, ...]` page array.
            shape: one_expert_page_shape(tensor_source),
            virtual_payload_offset: next_virtual_payload_offset,
            byte_count: tensor_byte_count,
        });
        source_intervals.push(QuantizedExpertSourceInterval {
            tensor_name: mangled_tensor_name,
            expert_start: 0,
            expert_count: 1,
            source_file_offset: next_segment_offset_bytes,
            source_byte_count: tensor_byte_count,
            virtual_payload_offset: next_virtual_payload_offset,
        });
        next_virtual_payload_offset += tensor_byte_count as u64;
        next_segment_offset_bytes +=
            align_up(tensor_byte_count as u64, PACK_SEGMENT_ALIGNMENT_BYTES);
    }
    Ok(QuantizedExpertShardManifest {
        source_file: expert_file_path.to_path_buf(),
        tensor_ranges,
        source_intervals,
        payload_byte_count: next_virtual_payload_offset,
    })
}

/// The loaded tensor name for one projection parameter, matching the shard
/// path's short page tensor identity.
fn expert_page_tensor_name(projection_name: &str, parameter_name: &str) -> String {
    format!("{projection_name}.{parameter_name}")
}

/// Page shape for one expert's slice: the expert-local dims behind a leading
/// page-slot axis of one.
fn one_expert_page_shape(tensor_source: &QuantizedTensorSource) -> Vec<usize> {
    let mut shape = vec![1_usize];
    shape.extend(expert_local_shape(tensor_source));
    shape
}

fn expert_local_shape(tensor_source: &QuantizedTensorSource) -> Vec<usize> {
    let mut shape = tensor_source.full_shape.clone();
    shape.remove(0);
    shape
}

/// Assembles slot-ordered per-expert arrays into the canonical tensor names the
/// paged-weight builder consumes. Concatenation stays lazy on the GPU; page
/// slot order must equal the normalized expert order used at manifest build
/// time so routed gathers address the correct expert.
#[cfg(feature = "direct-mlx")]
pub fn assemble_streaming_expert_page_tensors(
    runtime: &astronomical_runtime_integration::MlxRuntime,
    loaded_tensors: &mut HashMap<String, astronomical_runtime_integration::MlxArray>,
    layer_plan: &QuantizedExpertLayerPlan,
    expert_count: usize,
) -> Result<HashMap<String, astronomical_runtime_integration::MlxArray>, StreamingExpertPackError> {
    let mut assembled_tensors = HashMap::with_capacity(layer_plan.tensor_sources.len());
    for tensor_source in &layer_plan.tensor_sources {
        let canonical_tensor_name = expert_page_tensor_name(
            &tensor_source.projection_name,
            &tensor_source.parameter_name,
        );
        let mut slot_arrays = Vec::with_capacity(expert_count);
        for page_slot in 0..expert_count {
            let mangled_tensor_name = format!("{canonical_tensor_name}@{page_slot}");
            let loaded_array = loaded_tensors.remove(&mangled_tensor_name).ok_or(
                StreamingExpertPackError::MissingAssembledTensor {
                    canonical_tensor_name: canonical_tensor_name.clone(),
                    page_slot,
                },
            )?;
            slot_arrays.push(loaded_array);
        }
        let array_references = slot_arrays.iter().collect::<Vec<_>>();
        let concatenated_array =
            runtime
                .concatenate_axis(&array_references, 0)
                .map_err(|source| StreamingExpertPackError::ArrayConcatenation {
                    canonical_tensor_name: canonical_tensor_name.clone(),
                    description: source.to_string(),
                })?;
        assembled_tensors.insert(canonical_tensor_name, concatenated_array);
    }
    Ok(assembled_tensors)
}
