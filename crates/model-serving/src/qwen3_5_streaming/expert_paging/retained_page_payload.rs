//! Maps validated expert sources to the bytes retained after page decoding.

use std::mem::size_of;

use crate::expert_paging::QuantizedExpertLayerPlan;
use crate::qwen3_5_streaming::expert_paging::expert_pager::ExpertPagingError;

pub(crate) fn retained_expert_layer_page_payload_bytes(
    layer_plan: &QuantizedExpertLayerPlan,
    selected_expert_count: usize,
) -> Result<u64, ExpertPagingError> {
    let source_tensor_bytes_per_expert = layer_plan
        .tensor_sources
        .iter()
        .map(|tensor_source| tensor_source.bytes_per_expert)
        .collect::<Vec<_>>();
    let replaced_codebook_source_bytes_per_expert = layer_plan
        .tensor_sources
        .iter()
        .filter(|tensor_source| {
            matches!(
                tensor_source.parameter_name.as_str(),
                "tq2_packed" | "tq2_scales"
            )
        })
        .map(|tensor_source| tensor_source.bytes_per_expert)
        .collect::<Vec<_>>();
    let decoded_codebook_weight_shapes =
        layer_plan.jangtq2_decoded_weight_page_shapes(selected_expert_count)?;
    crate::memory::retained_expert_page_payload_bytes(
        &source_tensor_bytes_per_expert,
        &replaced_codebook_source_bytes_per_expert,
        selected_expert_count,
        &decoded_codebook_weight_shapes,
        size_of::<u16>(),
    )
    .ok_or_else(|| ExpertPagingError::InvalidPagingPlan {
        description: "retained expert-page payload byte count overflowed".to_owned(),
    })
}
