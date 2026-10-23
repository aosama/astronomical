//! Maps validated expert sources to the bytes one retained expert page holds.
//!
//! Every family retains its pages in source form: affine and native pages
//! were always kept as loaded, and jangtq2 codebook pages now stay packed
//! too (decoded bfloat16 cost eight times the bytes and kept the 122B
//! bundle's expert set from fitting in RAM). The retained payload is
//! therefore exactly the packed source bytes of the selected experts.

use crate::expert_paging::QuantizedExpertLayerPlan;
use crate::qwen3_5_streaming::expert_streaming::expert_pager::ExpertPagingError;

pub(crate) fn retained_expert_layer_page_payload_bytes(
    layer_plan: &QuantizedExpertLayerPlan,
    selected_expert_count: usize,
) -> Result<u64, ExpertPagingError> {
    crate::memory::source_expert_page_payload_bytes(
        layer_plan.expert_payload_byte_count()?,
        selected_expert_count,
    )
    .ok_or_else(|| ExpertPagingError::InvalidPagingPlan {
        description: "retained expert-page payload byte count overflowed".to_owned(),
    })
}
