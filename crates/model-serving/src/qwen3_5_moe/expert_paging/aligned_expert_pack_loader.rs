//! Startup activation and request-time loading of one complete prepared revision.

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use crate::{PerformanceAttribution, PerformanceCounter, PerformanceOperation};
use astronomical_runtime_integration::{
    MlxDtype, MlxMetalExpertPackLoadRange, MlxMetalExpertPackOutputTensor, MlxRuntime,
};

use super::{
    aligned_expert_pack::{
        AlignedExpertPackTensorDescriptor, read_aligned_expert_pack_header,
        validate_aligned_expert_pack_header,
    },
    expert_pager::{ExpertPagingError, PagedExpertWeights},
    paged_expert_weights::build_paged_expert_weights,
    quantized_expert_layer_plan::contiguous_selected_runs,
    quantized_expert_manifest::{QuantizedExpertLayerPlan, QuantizedTensorSource},
    safetensors_header::SafetensorsDtype,
};

#[derive(Debug)]
pub(super) struct AlignedExpertPackLayer {
    pack_path: PathBuf,
    tensor_descriptors: Vec<AlignedExpertPackTensorDescriptor>,
}

pub(super) fn discover_aligned_expert_pack_layers(
    model_directory: &Path,
    model_id: &str,
    model_revision: &str,
    layer_plans: &[QuantizedExpertLayerPlan],
) -> Vec<Option<AlignedExpertPackLayer>> {
    let revision_directory = model_directory
        .join(".astronomical-aligned-expert-packs")
        .join(model_revision);
    if !revision_directory.is_dir() {
        tracing::warn!(
            model_id,
            model_revision,
            revision_directory = %revision_directory.display(),
            "aligned expert-pack revision is absent; using bounded safetensors for every layer"
        );
        return empty_pack_layers(layer_plans.len());
    }
    let validated_layers = layer_plans
        .iter()
        .enumerate()
        .map(|(layer_index, layer_plan)| {
            let pack_path =
                revision_directory.join(format!("layer-{layer_index}.aligned-expert-pack"));
            let pack_header = read_aligned_expert_pack_header(&pack_path)?;
            validate_aligned_expert_pack_header(
                &pack_path,
                &pack_header,
                layer_plan,
                model_id,
                model_revision,
                layer_index,
            )?;
            Ok(AlignedExpertPackLayer {
                pack_path,
                tensor_descriptors: pack_header.tensor_descriptors,
            })
        })
        .collect::<Result<Vec<_>, super::aligned_expert_pack::AlignedExpertPackError>>();
    match validated_layers {
        Ok(validated_layers) => {
            let total_pack_byte_count = validated_layers
                .iter()
                .filter_map(|layer| std::fs::metadata(&layer.pack_path).ok())
                .map(|metadata| metadata.len())
                .fold(0_u64, u64::saturating_add);
            tracing::info!(
                model_id,
                model_revision,
                aligned_expert_pack_layer_count = validated_layers.len(),
                total_pack_byte_count,
                revision_directory = %revision_directory.display(),
                "activated complete aligned expert-pack revision"
            );
            validated_layers.into_iter().map(Some).collect()
        }
        Err(pack_error) => {
            tracing::warn!(
                model_id,
                model_revision,
                revision_directory = %revision_directory.display(),
                error = %pack_error,
                "aligned expert-pack revision is unavailable; using bounded safetensors for every layer"
            );
            empty_pack_layers(layer_plans.len())
        }
    }
}

pub(super) fn load_selected_experts_from_aligned_pack(
    runtime: &MlxRuntime,
    aligned_pack_layer: &AlignedExpertPackLayer,
    layer_plan: &QuantizedExpertLayerPlan,
    selected_expert_ids: &[usize],
) -> Result<PagedExpertWeights, ExpertPagingError> {
    let (metal_output_tensors, metal_io_load_ranges) =
        build_aligned_expert_pack_metal_io_descriptors(
            &aligned_pack_layer.tensor_descriptors,
            layer_plan,
            selected_expert_ids,
        )?;
    let metal_expert_pack_load = runtime
        .load_metal_expert_pack_ranges(
            &aligned_pack_layer.pack_path,
            &metal_output_tensors,
            &metal_io_load_ranges,
        )
        .map_err(|load_error| runtime_description(load_error.to_string()))?;
    let mut loaded_expert_tensors_by_name = HashMap::with_capacity(metal_output_tensors.len());
    for (output_tensor_index, tensor_descriptor) in
        aligned_pack_layer.tensor_descriptors.iter().enumerate()
    {
        let retained_output_array = metal_expert_pack_load
            .output_array(output_tensor_index)
            .map_err(|output_error| runtime_description(output_error.to_string()))?
            .retain()
            .map_err(|retain_error| runtime_description(retain_error.to_string()))?;
        loaded_expert_tensors_by_name.insert(
            format!(
                "{}.{}",
                tensor_descriptor.projection_name, tensor_descriptor.parameter_name
            ),
            retained_output_array,
        );
    }
    // Metal I/O is deliberately asynchronous. The native loader makes the model's GPU stream wait
    // on its completion event, so the CPU does not need to block before building the dependent
    // mixture-of-experts graph. Retaining this owner beside the page keeps the native transaction
    // and all destination buffers alive until the GPU has finished consuming them.
    let mut paged_expert_weights =
        build_paged_expert_weights(&mut loaded_expert_tensors_by_name, layer_plan)?;
    paged_expert_weights._metal_expert_pack_load_owner = Some(metal_expert_pack_load);
    Ok(paged_expert_weights)
}

/// Builds compact MLX output tensors and direct Metal I/O ranges for one expert page.
///
/// # Storage model
///
/// An aligned expert pack is tensor-major. Each tensor stores every expert consecutively:
/// expert 0 bytes, then expert 1 bytes, and so on. The destination is a compact tensor containing
/// only the router-selected experts in the same sorted order.
///
/// This gives adjacent selected expert IDs two useful properties at once:
///
/// - Their source bytes are adjacent in the pack file.
/// - Their destination page slots are adjacent in the compact MLX tensor.
///
/// Therefore one Metal I/O range can copy an entire adjacent run without changing any bytes or
/// adding a CPU copy. For example, selected IDs 4, 5, 6 become one three-expert transfer. The old
/// implementation submitted three transfers that described the same source and destination bytes.
/// Reducing those commands lowers CPU encoding work, especially when prompt processing selects
/// most or all experts in a layer.
///
/// # Why gaps remain separate
///
/// A selection such as 4, 5, 9 becomes two ranges. Combining it into one range would read experts
/// 6 through 8 even though the router did not select them. It would also write those unwanted bytes
/// into a compact destination that has no slots for them. We coalesce adjacency only; we never trade
/// extra payload or memory for fewer commands.
///
/// # Required invariants
///
/// Production supplies non-empty, strictly ascending, unique IDs within the layer's expert
/// capacity. That ordering is established before page construction and is also what makes every
/// run's first page slot unambiguous. The focused descriptor tests protect contiguous, scattered,
/// and mixed selections, including exact destination coverage and nonzero source starts.
pub fn build_aligned_expert_pack_metal_io_descriptors(
    tensor_descriptors: &[AlignedExpertPackTensorDescriptor],
    layer_plan: &QuantizedExpertLayerPlan,
    selected_expert_ids: &[usize],
) -> Result<
    (
        Vec<MlxMetalExpertPackOutputTensor>,
        Vec<MlxMetalExpertPackLoadRange>,
    ),
    ExpertPagingError,
> {
    // Compute the grouping once and reuse it for every tensor. Quantized affine expert layers have
    // weight, scale, and bias tensors for each projection; repeating this scan per tensor would add
    // avoidable request-path work and make it easier for tensor descriptors to diverge.
    let contiguous_selected_expert_runs = contiguous_selected_runs(selected_expert_ids);
    let mut output_tensors = Vec::with_capacity(tensor_descriptors.len());
    let mut metal_io_load_ranges =
        Vec::with_capacity(tensor_descriptors.len() * contiguous_selected_expert_runs.len());
    for (output_tensor_index, tensor_descriptor) in tensor_descriptors.iter().enumerate() {
        let tensor_source = tensor_source(layer_plan, &tensor_descriptor.tensor_name)?;
        // Only dimension zero changes. The output has one row per selected expert while every
        // expert's matrix geometry and dtype remain exactly as validated at startup.
        let mut selected_tensor_shape = Vec::with_capacity(tensor_source.full_shape.len());
        selected_tensor_shape.push(
            i32::try_from(selected_expert_ids.len())
                .map_err(|_| runtime_description("selected expert count exceeds i32"))?,
        );
        for dimension in &tensor_source.full_shape[1..] {
            selected_tensor_shape.push(
                i32::try_from(*dimension)
                    .map_err(|_| runtime_description("expert tensor dimension exceeds i32"))?,
            );
        }
        output_tensors.push(MlxMetalExpertPackOutputTensor::new(
            selected_tensor_shape,
            mlx_dtype(tensor_source.dtype)?,
        ));
        for (
            first_selected_expert_id,
            contiguous_selected_expert_count,
            first_selected_page_slot,
        ) in &contiguous_selected_expert_runs
        {
            // Source offset: locate the first selected expert inside this tensor's aligned segment.
            // Every expert has the same validated byte stride, so the complete run is contiguous.
            let source_file_offset_bytes = tensor_descriptor
                .pack_segment_offset_bytes
                .checked_add(
                    u64::try_from(*first_selected_expert_id)
                        .ok()
                        .and_then(|selected_expert_id| {
                            selected_expert_id
                                .checked_mul(tensor_descriptor.bytes_per_expert as u64)
                        })
                        .ok_or_else(|| runtime_description("expert source offset overflowed"))?,
                )
                .ok_or_else(|| runtime_description("expert source offset overflowed"))?;
            // Destination offset: page slots are dense even when global expert IDs have gaps.
            // A later source run starts immediately after the earlier selected experts in output.
            let output_tensor_offset_bytes = first_selected_page_slot
                .checked_mul(tensor_descriptor.bytes_per_expert)
                .ok_or_else(|| runtime_description("expert output offset overflowed"))?;
            // The byte count includes selected experts only. It never spans an unselected ID.
            let load_byte_count = contiguous_selected_expert_count
                .checked_mul(tensor_descriptor.bytes_per_expert)
                .ok_or_else(|| runtime_description("expert load range byte count overflowed"))?;
            metal_io_load_ranges.push(MlxMetalExpertPackLoadRange::new(
                output_tensor_index,
                output_tensor_offset_bytes,
                source_file_offset_bytes,
                load_byte_count,
            ));
        }
    }
    Ok((output_tensors, metal_io_load_ranges))
}

pub(super) fn load_selected_experts_from_aligned_pack_with_performance_attribution(
    runtime: &MlxRuntime,
    aligned_pack_layer: &AlignedExpertPackLayer,
    layer_plan: &QuantizedExpertLayerPlan,
    selected_expert_ids: &[usize],
    logical_payload_byte_count: u64,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<PagedExpertWeights, ExpertPagingError> {
    let paged_expert_weights = performance_attribution.measure_operation(
        PerformanceOperation::AlignedExpertPackMetalIoPageLoad,
        |_performance_attribution| {
            load_selected_experts_from_aligned_pack(
                runtime,
                aligned_pack_layer,
                layer_plan,
                selected_expert_ids,
            )
        },
    )?;
    performance_attribution
        .record_counter(PerformanceCounter::AlignedExpertPackMetalIoPageLoadCount, 1);
    performance_attribution.record_counter(
        PerformanceCounter::AlignedExpertPackMetalIoLogicalPayloadBytes,
        logical_payload_byte_count,
    );
    Ok(paged_expert_weights)
}

fn empty_pack_layers(layer_count: usize) -> Vec<Option<AlignedExpertPackLayer>> {
    (0..layer_count).map(|_| None).collect()
}

fn tensor_source<'plan>(
    layer_plan: &'plan QuantizedExpertLayerPlan,
    tensor_name: &str,
) -> Result<&'plan QuantizedTensorSource, ExpertPagingError> {
    layer_plan
        .tensor_sources
        .iter()
        .find(|source| source.tensor_name == tensor_name)
        .ok_or_else(|| runtime_description(format!("layer plan is missing {tensor_name:?}")))
}

fn mlx_dtype(dtype: SafetensorsDtype) -> Result<MlxDtype, ExpertPagingError> {
    match dtype {
        SafetensorsDtype::Uint32 => Ok(MlxDtype::UInt32),
        SafetensorsDtype::BFloat16 => Ok(MlxDtype::BFloat16),
        unsupported_dtype => Err(runtime_description(format!(
            "aligned expert loading does not support {unsupported_dtype}"
        ))),
    }
}

fn runtime_description(description: impl Into<String>) -> ExpertPagingError {
    ExpertPagingError::Runtime {
        description: description.into(),
    }
}
