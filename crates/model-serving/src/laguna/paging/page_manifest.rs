use std::collections::BTreeMap;
use std::path::PathBuf;

use crate::expert_paging::{
    QuantizedExpertPageManifest, QuantizedExpertShardManifest, QuantizedExpertSourceInterval,
    QuantizedExpertTensorRange, validate_expert_ids, validate_source_intervals,
    validate_virtual_intervals,
};

use super::error::LagunaPagingError;
use super::layer_plan::{LagunaPagedTensorSource, LagunaSparseLayerPagingPlan};

/// One shard accumulator used while packing selected expert intervals.
struct ShardPageBuilder {
    tensor_ranges: Vec<QuantizedExpertTensorRange>,
    source_intervals: Vec<QuantizedExpertSourceInterval>,
    virtual_payload_offset: u64,
}

/// Builds a compact page from already-resolved per-expert source intervals.
pub(super) fn build_laguna_expert_page_manifest(
    layer_plan: &LagunaSparseLayerPagingPlan,
    expert_ids: &[usize],
) -> Result<QuantizedExpertPageManifest, LagunaPagingError> {
    let normalized_expert_ids = validate_expert_ids(expert_ids, layer_plan.expert_capacity())?;
    let mut shard_builders: BTreeMap<PathBuf, ShardPageBuilder> = BTreeMap::new();
    for tensor_source in layer_plan.tensor_sources() {
        append_selected_tensor(
            tensor_source,
            &normalized_expert_ids,
            layer_plan.decoder_layer_index(),
            &mut shard_builders,
        )?;
    }

    let mut source_manifests = Vec::new();
    for (source_file, mut shard_builder) in shard_builders {
        shard_builder
            .source_intervals
            .sort_by_key(|interval| interval.source_file_offset);
        validate_source_intervals(&shard_builder.source_intervals, 0)?;
        validate_virtual_intervals(
            &shard_builder.source_intervals,
            shard_builder.virtual_payload_offset,
        )?;
        source_manifests.push(QuantizedExpertShardManifest {
            source_file,
            tensor_ranges: shard_builder.tensor_ranges,
            source_intervals: shard_builder.source_intervals,
            payload_byte_count: shard_builder.virtual_payload_offset,
        });
    }

    let payload_byte_count = source_manifests
        .iter()
        .try_fold(0_u64, |total, manifest| {
            total.checked_add(manifest.payload_byte_count)
        })
        .ok_or(LagunaPagingError::ExpertPayloadOverflow {
            layer_index: layer_plan.decoder_layer_index(),
        })?;
    Ok(QuantizedExpertPageManifest {
        expert_ids: normalized_expert_ids.clone(),
        page_slot_by_global_expert_id: page_slot_lookup(
            &normalized_expert_ids,
            layer_plan.expert_capacity(),
        )?,
        source_manifests,
        payload_byte_count,
    })
}

fn append_selected_tensor(
    tensor_source: &LagunaPagedTensorSource,
    selected_expert_ids: &[usize],
    decoder_layer_index: usize,
    shard_builders: &mut BTreeMap<PathBuf, ShardPageBuilder>,
) -> Result<(), LagunaPagingError> {
    // Experts from one canonical tensor may live in different shards.
    let mut selected_by_file: BTreeMap<PathBuf, Vec<usize>> = BTreeMap::new();
    for expert_id in selected_expert_ids {
        let slice = tensor_source.expert_slices.get(*expert_id).ok_or(
            LagunaPagingError::MissingPerExpertSource {
                tensor_id: tensor_source.tensor_id,
                expert_index: *expert_id,
            },
        )?;
        selected_by_file
            .entry(slice.source_file.clone())
            .or_default()
            .push(*expert_id);
    }
    for (source_file, file_expert_ids) in selected_by_file {
        let shard_builder = shard_builders
            .entry(source_file)
            .or_insert(ShardPageBuilder {
                tensor_ranges: Vec::new(),
                source_intervals: Vec::new(),
                virtual_payload_offset: 0,
            });
        append_file_tensor_run(
            tensor_source,
            &file_expert_ids,
            decoder_layer_index,
            shard_builder,
        )?;
    }
    Ok(())
}

fn append_file_tensor_run(
    tensor_source: &LagunaPagedTensorSource,
    file_expert_ids: &[usize],
    decoder_layer_index: usize,
    shard_builder: &mut ShardPageBuilder,
) -> Result<(), LagunaPagingError> {
    let selected_byte_count = usize::try_from(selected_tensor_byte_count(
        tensor_source,
        file_expert_ids.len(),
        decoder_layer_index,
    )?)
    .map_err(|_| LagunaPagingError::ExpertPayloadOverflow {
        layer_index: decoder_layer_index,
    })?;
    let mut compact_shape = vec![file_expert_ids.len()];
    compact_shape.extend(&tensor_source.compact_trailing_shape);
    shard_builder
        .tensor_ranges
        .push(QuantizedExpertTensorRange {
            tensor_name: compact_tensor_name(tensor_source),
            projection_name: tensor_source.projection_name.to_owned(),
            parameter_name: tensor_source.parameter_name.to_owned(),
            dtype: tensor_source.dtype,
            shape: compact_shape,
            virtual_payload_offset: shard_builder.virtual_payload_offset,
            byte_count: selected_byte_count,
        });

    let mut run_start_index = 0;
    while run_start_index < file_expert_ids.len() {
        let first_expert_id = file_expert_ids[run_start_index];
        let first_slice = &tensor_source.expert_slices[first_expert_id];
        let mut run_expert_count = 1;
        let mut expected_next_offset = first_slice
            .source_file_offset
            .checked_add(u64::try_from(first_slice.source_byte_count).map_err(|_| {
                LagunaPagingError::ExpertPayloadOverflow {
                    layer_index: decoder_layer_index,
                }
            })?)
            .ok_or(LagunaPagingError::ExpertPayloadOverflow {
                layer_index: decoder_layer_index,
            })?;
        while run_start_index + run_expert_count < file_expert_ids.len() {
            let next_expert_id = file_expert_ids[run_start_index + run_expert_count];
            if next_expert_id != first_expert_id + run_expert_count {
                break;
            }
            let next_slice = &tensor_source.expert_slices[next_expert_id];
            if next_slice.source_file_offset != expected_next_offset {
                break;
            }
            expected_next_offset = expected_next_offset
                .checked_add(u64::try_from(next_slice.source_byte_count).map_err(|_| {
                    LagunaPagingError::ExpertPayloadOverflow {
                        layer_index: decoder_layer_index,
                    }
                })?)
                .ok_or(LagunaPagingError::ExpertPayloadOverflow {
                    layer_index: decoder_layer_index,
                })?;
            run_expert_count += 1;
        }
        let run_byte_count = first_slice
            .source_byte_count
            .checked_mul(run_expert_count)
            .ok_or(LagunaPagingError::ExpertPayloadOverflow {
                layer_index: decoder_layer_index,
            })?;
        let run_virtual_offset = shard_builder
            .virtual_payload_offset
            .checked_add(
                u64::try_from(run_start_index * tensor_source.bytes_per_expert).map_err(|_| {
                    LagunaPagingError::ExpertPayloadOverflow {
                        layer_index: decoder_layer_index,
                    }
                })?,
            )
            .ok_or(LagunaPagingError::ExpertPayloadOverflow {
                layer_index: decoder_layer_index,
            })?;
        shard_builder
            .source_intervals
            .push(QuantizedExpertSourceInterval {
                tensor_name: compact_tensor_name(tensor_source),
                expert_start: first_expert_id,
                expert_count: run_expert_count,
                source_file_offset: first_slice.source_file_offset,
                source_byte_count: run_byte_count,
                virtual_payload_offset: run_virtual_offset,
            });
        run_start_index += run_expert_count;
    }
    shard_builder.virtual_payload_offset = shard_builder
        .virtual_payload_offset
        .checked_add(u64::try_from(selected_byte_count).map_err(|_| {
            LagunaPagingError::ExpertPayloadOverflow {
                layer_index: decoder_layer_index,
            }
        })?)
        .ok_or(LagunaPagingError::ExpertPayloadOverflow {
            layer_index: decoder_layer_index,
        })?;
    Ok(())
}

fn compact_tensor_name(tensor_source: &LagunaPagedTensorSource) -> String {
    format!(
        "{}.{}",
        tensor_source.projection_name, tensor_source.parameter_name
    )
}

fn selected_tensor_byte_count(
    tensor_source: &LagunaPagedTensorSource,
    selected_expert_count: usize,
    decoder_layer_index: usize,
) -> Result<u64, LagunaPagingError> {
    u64::try_from(tensor_source.bytes_per_expert)
        .ok()
        .and_then(|bytes_per_expert| {
            bytes_per_expert.checked_mul(u64::try_from(selected_expert_count).ok()?)
        })
        .ok_or(LagunaPagingError::ExpertPayloadOverflow {
            layer_index: decoder_layer_index,
        })
}

fn page_slot_lookup(
    normalized_expert_ids: &[usize],
    expert_capacity: usize,
) -> Result<Vec<u32>, LagunaPagingError> {
    let mut page_slot_by_global_expert_id = vec![u32::MAX; expert_capacity];
    for (page_slot, expert_id) in normalized_expert_ids.iter().copied().enumerate() {
        page_slot_by_global_expert_id[expert_id] = u32::try_from(page_slot).map_err(|_| {
            crate::expert_paging::ExpertManifestError::PageSlotExceedsU32 { page_slot }
        })?;
    }
    Ok(page_slot_by_global_expert_id)
}
