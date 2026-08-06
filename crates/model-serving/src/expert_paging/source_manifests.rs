use super::quantized_expert_manifest::{
    ExpertManifestError, QuantizedExpertShardManifest, QuantizedExpertSourceInterval,
    QuantizedExpertTensorRange, QuantizedTensorSource,
};
use super::quantized_expert_validation::{validate_source_intervals, validate_virtual_intervals};
use std::collections::BTreeSet;

/// Groups source intervals by shard file and builds compact shard manifests.
pub fn build_source_manifests(
    tensor_sources: &[QuantizedTensorSource],
    selected_expert_ids: &[usize],
) -> Result<Vec<QuantizedExpertShardManifest>, ExpertManifestError> {
    let source_files = tensor_sources
        .iter()
        .map(|tensor_source| tensor_source.source_file.clone())
        .collect::<BTreeSet<_>>();
    let mut source_manifests = Vec::with_capacity(source_files.len());

    for source_file in source_files {
        let mut tensor_ranges = Vec::new();
        let mut source_intervals = Vec::new();
        let mut virtual_payload_offset = 0_u64;

        for tensor_source in tensor_sources {
            if tensor_source.source_file != source_file {
                continue;
            }
            let selected_tensor_byte_count =
                selected_expert_ids.len() * tensor_source.bytes_per_expert;
            tensor_ranges.push(QuantizedExpertTensorRange {
                tensor_name: format!(
                    "{}.{}",
                    tensor_source.projection_name, tensor_source.parameter_name
                ),
                projection_name: tensor_source.projection_name.clone(),
                parameter_name: tensor_source.parameter_name.clone(),
                dtype: tensor_source.dtype,
                shape: {
                    let mut shape = vec![selected_expert_ids.len()];
                    shape.extend(&tensor_source.full_shape[1..]);
                    shape
                },
                virtual_payload_offset,
                byte_count: selected_tensor_byte_count,
            });
            for (expert_start, expert_count, first_page_slot) in
                contiguous_selected_runs(selected_expert_ids)
            {
                source_intervals.push(QuantizedExpertSourceInterval {
                    tensor_name: tensor_source.tensor_name.clone(),
                    expert_start,
                    expert_count,
                    source_file_offset: tensor_source.tensor_payload_offset
                        + expert_start as u64 * tensor_source.bytes_per_expert as u64,
                    source_byte_count: expert_count * tensor_source.bytes_per_expert,
                    virtual_payload_offset: virtual_payload_offset
                        + first_page_slot as u64 * tensor_source.bytes_per_expert as u64,
                });
            }
            virtual_payload_offset += selected_tensor_byte_count as u64;
        }

        let mut ordered_source_intervals = source_intervals;
        ordered_source_intervals.sort_by_key(|interval| interval.source_file_offset);
        validate_source_intervals(&ordered_source_intervals, 0)?;
        validate_virtual_intervals(&ordered_source_intervals, virtual_payload_offset)?;
        source_manifests.push(QuantizedExpertShardManifest {
            source_file,
            tensor_ranges,
            source_intervals: ordered_source_intervals,
            payload_byte_count: virtual_payload_offset,
        });
    }
    Ok(source_manifests)
}

/// Groups adjacent selected experts while retaining their first page slot.
pub fn contiguous_selected_runs(expert_ids: &[usize]) -> Vec<(usize, usize, usize)> {
    let mut runs = Vec::new();
    if expert_ids.is_empty() {
        return runs;
    }
    let mut run_start = expert_ids[0];
    let mut first_page_slot = 0;
    for page_slot in 1..=expert_ids.len() {
        let run_is_complete = page_slot == expert_ids.len();
        if !run_is_complete && expert_ids[page_slot] == expert_ids[page_slot - 1] + 1 {
            continue;
        }
        let run_end = expert_ids[page_slot - 1];
        runs.push((run_start, run_end - run_start + 1, first_page_slot));
        if !run_is_complete {
            run_start = expert_ids[page_slot];
            first_page_slot = page_slot;
        }
    }
    runs
}
