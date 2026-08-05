//! Experimental direct Metal input/output descriptor construction.

use astronomical_model_serving::{
    ExpertPagingError, QuantizedExpertLayerPlan, QuantizedTensorSource, SafetensorsDtype,
    contiguous_selected_runs,
};
use astronomical_runtime_integration::{
    MlxDtype, MlxMetalExpertPackLoadRange, MlxMetalExpertPackOutputTensor,
};

use crate::AlignedExpertPackTensorDescriptor;

/// Builds compact MLX output tensors and direct Metal input/output ranges for one expert page.
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
    let contiguous_selected_expert_runs = contiguous_selected_runs(selected_expert_ids);
    let mut output_tensors = Vec::with_capacity(tensor_descriptors.len());
    let mut metal_io_load_ranges =
        Vec::with_capacity(tensor_descriptors.len() * contiguous_selected_expert_runs.len());
    for (output_tensor_index, tensor_descriptor) in tensor_descriptors.iter().enumerate() {
        let tensor_source = tensor_source(layer_plan, &tensor_descriptor.tensor_name)?;
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
            let output_tensor_offset_bytes = first_selected_page_slot
                .checked_mul(tensor_descriptor.bytes_per_expert)
                .ok_or_else(|| runtime_description("expert output offset overflowed"))?;
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

fn tensor_source<'plan>(
    layer_plan: &'plan QuantizedExpertLayerPlan,
    tensor_name: &str,
) -> Result<&'plan QuantizedTensorSource, ExpertPagingError> {
    layer_plan
        .tensor_sources
        .iter()
        .find(|tensor_source| tensor_source.tensor_name == tensor_name)
        .ok_or_else(|| runtime_description(format!("layer plan is missing {tensor_name:?}")))
}

fn mlx_dtype(dtype: SafetensorsDtype) -> Result<MlxDtype, ExpertPagingError> {
    match dtype {
        SafetensorsDtype::Uint32 => Ok(MlxDtype::UInt32),
        SafetensorsDtype::Float16 => Ok(MlxDtype::Float16),
        SafetensorsDtype::BFloat16 => Ok(MlxDtype::BFloat16),
        SafetensorsDtype::Float32 => Ok(MlxDtype::Float32),
        unsupported_dtype => Err(runtime_description(format!(
            "experimental aligned expert loading does not support {unsupported_dtype}"
        ))),
    }
}

fn runtime_description(description: impl Into<String>) -> ExpertPagingError {
    ExpertPagingError::Runtime {
        description: description.into(),
    }
}
