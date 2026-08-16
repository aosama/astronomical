use thiserror::Error;

use super::tensor_id::{LagunaExpertProjection, LagunaTensorComponent, LagunaTensorId};

/// Typed ambiguity, bounds, collision, or completeness failure for Laguna names.
#[derive(Debug, Eq, Error, PartialEq)]
pub enum LagunaTensorNameNormalizationError {
    #[error("Laguna tensor normalization requires at least one decoder layer")]
    InvalidLayerCount,
    #[error(
        "Laguna tensor inventory has {actual_count} names, exceeding the {maximum_count}-name limit"
    )]
    TensorInventoryTooLarge {
        actual_count: usize,
        maximum_count: usize,
    },
    #[error(
        "Laguna tensor name has {actual_bytes} bytes, exceeding the {maximum_bytes}-byte limit"
    )]
    TensorNameTooLong {
        actual_bytes: usize,
        maximum_bytes: usize,
    },
    #[error("Laguna tensor name must not be empty")]
    EmptyTensorName,
    #[error("Laguna tensor name '{tensor_name}' repeats the language_model wrapper")]
    RepeatedLanguageModelWrapper { tensor_name: String },
    #[error("Laguna artifact mixes bare and language_model-wrapped tensor namespaces")]
    MixedTensorNamespaces,
    #[error("Laguna tensor name '{tensor_name}' does not begin with model or lm_head")]
    UnknownTensorRoot { tensor_name: String },
    #[error("unknown executable Laguna tensor name '{tensor_name}'")]
    UnknownTensorName { tensor_name: String },
    #[error(
        "Laguna tensor '{tensor_name}' uses layer {layer_index}, outside layer count {layer_count}"
    )]
    InvalidLayerIndex {
        tensor_name: String,
        layer_index: usize,
        layer_count: usize,
    },
    #[error(
        "Laguna tensor '{tensor_name}' uses expert {expert_index}, outside expert count {expert_count}"
    )]
    InvalidExpertIndex {
        tensor_name: String,
        expert_index: usize,
        expert_count: usize,
    },
    #[error("Laguna layer {layer_index} declares routed-expert tensors but expert count is zero")]
    ExpertTensorWithoutExperts { layer_index: usize },
    #[error(
        "Laguna sources '{first_source_name}' and '{conflicting_source_name}' collide at {tensor_id:?}"
    )]
    CanonicalCollision {
        tensor_id: LagunaTensorId,
        first_source_name: String,
        conflicting_source_name: String,
    },
    #[error(
        "Laguna layer {layer_index} {projection:?} {component:?} has {actual_expert_count} experts, expected {expected_expert_count}"
    )]
    IncompleteExpertSet {
        layer_index: usize,
        projection: LagunaExpertProjection,
        component: LagunaTensorComponent,
        expected_expert_count: usize,
        actual_expert_count: usize,
    },
    #[error("Laguna layer {layer_index} has a partial routed-expert {component:?} projection set")]
    IncompleteExpertProjectionSet {
        layer_index: usize,
        component: LagunaTensorComponent,
    },
    #[error("Laguna layer {layer_index} mixes stacked and per-expert source packaging")]
    MixedExpertPackaging { layer_index: usize },
    #[error("Laguna layer {layer_index} mixes fused and split expert gate/up layouts")]
    MixedExpertGateUpLayouts { layer_index: usize },
}
