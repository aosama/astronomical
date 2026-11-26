//! What the Qwen3.5 loader derives for the engine (issue #2473): the
//! tensor-name maps the expert plans read, the `astro_decoder_state_cache`
//! layout, and the memory facts the RAM owner plans from.

use std::collections::HashMap;

use runtime_integration::MlxRuntime;

use crate::qwen3_5_core::artifacts::{QuantizedExpertLayerPlan, Qwen3_5ShardIndex};
use crate::qwen3_5_core::configuration::Qwen3_5Config;
use crate::qwen3_5_core::model::bound_weight_dtype_flow::derive_astro_decoder_state_cache_layer_dtypes;
use crate::qwen3_5_core::model_math::weights::Qwen3_5Weights;
use crate::{
    AstroDecoderStateCacheLayout, ModelMemoryFacts, ModelMemoryGeometry, PerformanceAttribution,
    PerformanceOperation, TensorInventory,
};
use text_generation::execution_error::TextExecutionError;
use text_generation::model::model_chunking_configuration::TextModelChunkingConfiguration;

/// The lookups an expert plan needs to find each expert tensor on disk.
pub struct ExpertTensorNameMaps {
    pub tensor_name_to_shard_file_name: HashMap<String, String>,
    pub stored_tensor_name_by_canonical_name: HashMap<String, String>,
}

/// Reads both lookups from the validated shard index and tensor inventory.
#[must_use]
pub fn expert_tensor_name_maps(
    shard_index: &Qwen3_5ShardIndex,
    tensor_inventory: &TensorInventory,
) -> ExpertTensorNameMaps {
    let tensor_name_to_shard_file_name = shard_index
        .language_tensor_name_to_shard_file_name()
        .iter()
        .map(|(tensor_name, shard_file_name)| (tensor_name.clone(), shard_file_name.clone()))
        .collect();
    let stored_tensor_name_by_canonical_name = tensor_inventory
        .locations()
        .map(|location| {
            (
                location.canonical_name().to_owned(),
                location.stored_name().to_owned(),
            )
        })
        .collect();
    ExpertTensorNameMaps {
        tensor_name_to_shard_file_name,
        stored_tensor_name_by_canonical_name,
    }
}

/// The expert payload a loader measured, in the units the RAM owner plans by.
pub struct ExpertPayloadGeometry {
    pub complete_expert_payload_bytes: u64,
    pub largest_complete_expert_layer_bytes: u64,
    pub largest_routed_expert_page_bytes: u64,
}

/// The memory facts of a loaded model under the runtime's current MLX
/// memory ceiling.
pub fn model_memory_facts(
    runtime: &MlxRuntime,
    model_core_payload_bytes: u64,
    expert_payload_geometry: ExpertPayloadGeometry,
    astro_decoder_state_cache_layout: &AstroDecoderStateCacheLayout,
) -> Result<ModelMemoryFacts, TextExecutionError> {
    let active_memory_limit_bytes =
        u64::try_from(runtime.memory_limits().active_memory_limit_bytes()).map_err(|_| {
            TextExecutionError::InvalidInput {
                description: "MLX active memory ceiling exceeds the u64 range",
            }
        })?;
    let sequence_state_bytes_per_token = u64::try_from(
        astro_decoder_state_cache_layout
            .sequence_state_payload_byte_count_per_token()
            .unwrap_or(0),
    )
    .unwrap_or(0);
    ModelMemoryFacts::new(
        active_memory_limit_bytes,
        ModelMemoryGeometry {
            model_core_payload_bytes,
            complete_expert_payload_bytes: expert_payload_geometry.complete_expert_payload_bytes,
            largest_complete_expert_layer_bytes: expert_payload_geometry
                .largest_complete_expert_layer_bytes,
            largest_routed_expert_page_bytes: expert_payload_geometry
                .largest_routed_expert_page_bytes,
            sequence_state_bytes_per_token,
        },
    )
    .map_err(|_| TextExecutionError::InvalidInput {
        description: "the MLX memory limit must be positive",
    })
}

/// Derives the `astro_decoder_state_cache` layout under the tensor-binding
/// attribution span.
///
/// The layout cannot be finalized before weights and sparse plans are bound:
/// their source dtypes decide MLX result-type promotion, so the persistence
/// contract is derived at this boundary without evaluating tensors. A sparse
/// layer with no plan here fails the derivation with the shared "no layer
/// plan" text.
pub fn measured_astro_decoder_state_cache_layout(
    config: &Qwen3_5Config,
    chunking: &TextModelChunkingConfiguration,
    weights: &Qwen3_5Weights,
    expert_layer_plans: &[QuantizedExpertLayerPlan],
    performance_attribution: &mut PerformanceAttribution,
) -> Result<AstroDecoderStateCacheLayout, TextExecutionError> {
    performance_attribution.measure_operation(
        PerformanceOperation::ModelTensorBinding,
        |_performance_attribution| {
            astro_decoder_state_cache_layout_at_tensor_binding(
                config,
                chunking,
                weights,
                expert_layer_plans,
            )
        },
    )
}

fn astro_decoder_state_cache_layout_at_tensor_binding(
    config: &Qwen3_5Config,
    chunking: &TextModelChunkingConfiguration,
    weights: &Qwen3_5Weights,
    expert_layer_plans: &[QuantizedExpertLayerPlan],
) -> Result<AstroDecoderStateCacheLayout, TextExecutionError> {
    let astro_decoder_state_cache_layer_dtypes =
        derive_astro_decoder_state_cache_layer_dtypes(weights, expert_layer_plans)?;
    let full_attention_key_value_growth_tokens =
        usize::try_from(chunking.full_attention_key_value_growth_tokens).map_err(|_| {
            TextExecutionError::InvalidInput {
                description: "full-attention key/value growth tokens exceed the usize range",
            }
        })?;
    crate::qwen3_5_core::decoder::qwen3_5_astro_decoder_state_cache_layout(
        config,
        full_attention_key_value_growth_tokens,
        &astro_decoder_state_cache_layer_dtypes,
    )
    .map_err(|astro_decoder_state_cache_layout_error| {
        TextExecutionError::InvalidAstroDecoderStateCacheLayout {
            description: astro_decoder_state_cache_layout_error.to_string(),
        }
    })
}
