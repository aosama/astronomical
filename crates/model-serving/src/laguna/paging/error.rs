use thiserror::Error;

use crate::expert_paging::ExpertManifestError;
use crate::laguna::artifacts::LagunaTensorId;

/// Fail-closed Laguna paging construction from canonical descriptors.
#[derive(Debug, Error)]
pub enum LagunaPagingError {
    #[error("Laguna sparse layer {layer_index} is missing canonical routed tensor {tensor_id:?}")]
    MissingRoutedExpertTensor {
        layer_index: usize,
        tensor_id: LagunaTensorId,
    },
    #[error("Laguna routed tensor {tensor_id:?} uses storage that this pager does not execute")]
    UnsupportedRoutedStorage { tensor_id: LagunaTensorId },
    #[error("Laguna routed tensor {tensor_id:?} has no retained source interval")]
    MissingSourceInterval { tensor_id: LagunaTensorId },
    #[error(
        "Laguna routed tensor {tensor_id:?} expert {expert_index} is outside the retained source list"
    )]
    MissingPerExpertSource {
        tensor_id: LagunaTensorId,
        expert_index: usize,
    },
    #[error("Laguna routed tensor {tensor_id:?} payload is not divisible by the expert count")]
    ExpertPayloadNotDivisible { tensor_id: LagunaTensorId },
    #[error("Laguna routed tensor {tensor_id:?} fused source is not an even gate/up split")]
    FusedProjectionNotEven { tensor_id: LagunaTensorId },
    #[error("Laguna sparse layer {layer_index} expert payload accounting overflowed")]
    ExpertPayloadOverflow { layer_index: usize },
    #[error(
        "Laguna sparse layer {layer_index} complete payload {complete_payload_bytes} disagrees with {expert_payload_bytes} times {expert_capacity}"
    )]
    InconsistentCompletePayload {
        layer_index: usize,
        complete_payload_bytes: u64,
        expert_payload_bytes: u64,
        expert_capacity: usize,
    },
    #[error("Laguna routed tensor {tensor_id:?} uses an unsupported SafeTensors dtype")]
    UnsupportedSourceDtype { tensor_id: LagunaTensorId },
    #[error("Laguna page manifest construction failed")]
    Manifest(#[from] ExpertManifestError),
    #[error("Laguna streamed page is missing tensor {tensor_name}")]
    MissingPagedTensor { tensor_name: String },
    #[error("Laguna streamed page execution failed: {description}")]
    PageExecution { description: &'static str },
    #[error("Laguna sliding-window prefill transient is invalid")]
    InvalidSlidingTransient,
}
