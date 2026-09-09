//! Direct Metal input/output descriptors for one per-expert pack file.

use astronomical_model_serving::ExpertPagingError;
use astronomical_runtime_integration::{
    MlxDtype, MlxMetalExpertPackLoadRange, MlxMetalExpertPackOutputTensor,
};

use crate::per_expert_pack::PerExpertPackHeader;

/// Builds compact output tensors and one load range per tensor for a single expert file.
pub fn build_per_expert_pack_metal_io_descriptors(
    per_expert_pack_header: &PerExpertPackHeader,
    page_slot: usize,
) -> Result<
    (
        Vec<MlxMetalExpertPackOutputTensor>,
        Vec<MlxMetalExpertPackLoadRange>,
    ),
    ExpertPagingError,
> {
    let mut output_tensors = Vec::with_capacity(per_expert_pack_header.tensor_descriptors.len());
    let mut metal_io_load_ranges =
        Vec::with_capacity(per_expert_pack_header.tensor_descriptors.len());
    for (output_tensor_index, tensor_descriptor) in
        per_expert_pack_header.tensor_descriptors.iter().enumerate()
    {
        let mut selected_tensor_shape =
            Vec::with_capacity(tensor_descriptor.expert_local_shape.len() + 1);
        selected_tensor_shape.push(1);
        for dimension in &tensor_descriptor.expert_local_shape {
            selected_tensor_shape.push(
                i32::try_from(*dimension)
                    .map_err(|_| runtime_description("expert tensor dimension exceeds i32"))?,
            );
        }
        output_tensors.push(MlxMetalExpertPackOutputTensor::new(
            selected_tensor_shape,
            mlx_dtype_name(&tensor_descriptor.dtype_name)?,
        ));
        let output_tensor_offset_bytes = page_slot
            .checked_mul(tensor_descriptor.bytes_per_expert)
            .ok_or_else(|| runtime_description("expert output offset overflowed"))?;
        metal_io_load_ranges.push(MlxMetalExpertPackLoadRange::new(
            output_tensor_index,
            output_tensor_offset_bytes,
            tensor_descriptor.pack_segment_offset_bytes,
            tensor_descriptor.bytes_per_expert,
        ));
    }
    Ok((output_tensors, metal_io_load_ranges))
}

fn mlx_dtype_name(dtype_name: &str) -> Result<MlxDtype, ExpertPagingError> {
    match dtype_name {
        "U32" | "uint32" => Ok(MlxDtype::UInt32),
        "F16" | "float16" => Ok(MlxDtype::Float16),
        "BF16" | "bfloat16" => Ok(MlxDtype::BFloat16),
        "F32" | "float32" => Ok(MlxDtype::Float32),
        unsupported_dtype => Err(runtime_description(format!(
            "experimental per-expert loading does not support {unsupported_dtype}"
        ))),
    }
}

fn runtime_description(description: impl Into<String>) -> ExpertPagingError {
    ExpertPagingError::InvalidPagingPlan {
        description: description.into(),
    }
}
