//! Native Rust owner for the K2 Horizon MoVA family.
//!
//! See `agents.md` in this directory for memory ownership and family-not-artifact rules.

mod artifacts;
mod cache_layout;
mod configuration;
mod expert_geometry;
mod text;

#[cfg(feature = "direct-mlx")]
mod engine;
#[cfg(feature = "direct-mlx")]
mod model;
mod serving_settings;
#[cfg(feature = "direct-mlx")]
mod startup;

pub use artifacts::{
    K2HorizonMoVAArtifactValidationError, K2HorizonMoVAArtifactValidator, K2HorizonMoVAShardIndex,
    K2HorizonMoVAWeightDialect, ValidatedK2HorizonMoVAArtifact,
    expected_stacked_affine_tensor_names,
};
pub use cache_layout::k2_horizon_mova_decoder_cache_layout;
pub use configuration::{
    K2HorizonMoVAAffineProfile, K2HorizonMoVAAttentionGateFunc, K2HorizonMoVAConfig,
    K2HorizonMoVAConfigError, K2HorizonMoVALayerKind, K2HorizonMoVAQuantizationContract,
};
pub use expert_geometry::{
    K2HorizonMoVAExpertGeometryError, K2HorizonMoVASparseLayerExpertPayload,
    k2_horizon_mova_expert_layer_geometries,
};
pub use text::{
    K2HorizonMoVAGenerationProcessor, K2HorizonMoVAInferenceRequest, K2HorizonMoVAOutputParser,
    K2HorizonMoVAPromptRenderer, K2HorizonMoVARequestOutput, K2HorizonMoVAThinkingBudgetError,
    K2HorizonMoVAThinkingBudgetState, K2HorizonMoVATokenizer, K2HorizonMoVATokenizerError,
    resolve_k2_horizon_mova_thinking_budget,
};

#[cfg(feature = "direct-mlx")]
pub use engine::{K2HorizonMoVAEngine, K2HorizonMoVAInferenceExecution};
#[cfg(feature = "direct-mlx")]
pub use model::{
    FusedExpertDecodeKernels, K2HorizonMoVAAffineLinear, gathered_fused_swiglu,
    gathered_value_experts,
};
pub use serving_settings::K2HorizonMoVAServingSettings;
#[cfg(feature = "direct-mlx")]
pub use startup::{
    K2HorizonMoVAStartupError, initialize_k2_horizon_mova_execution,
    initialize_k2_horizon_mova_execution_with_serving_settings, initialize_k2_horizon_mova_model,
    initialize_k2_horizon_mova_model_with_serving_settings,
};
