//! Shared synthetic Laguna fixtures for direct-MLX decoder-state journeys.
//!
//! One tiny mixed contract (a full-attention append-only layer followed by a
//! sliding-window rotating layer) keeps capture/restore journeys cheap while
//! exercising both persistent state kinds in one synthetic, non-pinned layer
//! ordering. The fixture is intentionally independent of any downloaded model
//! artifact so mismatch-rejection proofs never require extra downloads.

use std::collections::HashMap;

use astronomical_model_serving::{
    LagunaAttentionProjection, LagunaExpertProjection, LagunaGlobalTensorRole,
    LagunaLayerTensorRole, LagunaNativeWeights, LagunaTargetContract, LagunaTargetNormalizer,
    LagunaTensorComponent, LagunaTensorId,
};
use astronomical_runtime_integration::{MlxArray, MlxRuntime};
use serde_json::json;

/// A two-layer Laguna contract: layer 0 is full attention (append-only state)
/// and layer 1 is sliding-window attention (rotating state with window 4).
pub(crate) fn tiny_mixed_contract() -> LagunaTargetContract {
    let config = json!({
        "architectures": ["LagunaForCausalLM"],
        "model_type": "laguna",
        "vocab_size": 8,
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 2,
        "max_position_embeddings": 32,
        "rms_norm_eps": 0.00001,
        "tie_word_embeddings": false,
        "torch_dtype": "float32",
        "layer_types": ["full", "sliding"],
        "sliding_window": 4,
        "mlp_layer_types": ["dense", "dense"],
        "gating_types": ["per_head", "none"],
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 1.0
        }
    });
    LagunaTargetNormalizer::normalize(&serde_json::to_vec(&config).expect("config bytes"))
        .expect("tiny mixed Laguna contract should normalize")
}

/// Binds deterministic synthetic weights shaped by the tiny mixed contract.
pub(crate) fn bind_tiny_weights(
    runtime: &MlxRuntime,
    contract: &LagunaTargetContract,
) -> LagunaNativeWeights {
    let mut tensors = HashMap::new();
    tensors.insert(
        weight_id(LagunaGlobalTensorRole::TokenEmbedding),
        ones(runtime, &[8, 8]),
    );
    tensors.insert(
        weight_id(LagunaGlobalTensorRole::FinalNormalization),
        ones(runtime, &[8]),
    );
    tensors.insert(
        weight_id(LagunaGlobalTensorRole::OutputHead),
        ones(runtime, &[8, 8]),
    );
    for layer_index in 0..2 {
        tensors.insert(
            layer_weight_id(layer_index, LagunaLayerTensorRole::InputNormalization),
            ones(runtime, &[8]),
        );
        tensors.insert(
            layer_weight_id(
                layer_index,
                LagunaLayerTensorRole::PostAttentionNormalization,
            ),
            ones(runtime, &[8]),
        );
        tensors.insert(
            layer_weight_id(
                layer_index,
                LagunaLayerTensorRole::Attention(LagunaAttentionProjection::Query),
            ),
            ones(runtime, &[8, 8]),
        );
        tensors.insert(
            layer_weight_id(
                layer_index,
                LagunaLayerTensorRole::Attention(LagunaAttentionProjection::Key),
            ),
            ones(runtime, &[4, 8]),
        );
        tensors.insert(
            layer_weight_id(
                layer_index,
                LagunaLayerTensorRole::Attention(LagunaAttentionProjection::Value),
            ),
            ones(runtime, &[4, 8]),
        );
        tensors.insert(
            layer_weight_id(
                layer_index,
                LagunaLayerTensorRole::Attention(LagunaAttentionProjection::Output),
            ),
            ones(runtime, &[8, 8]),
        );
        tensors.insert(
            layer_weight_id(
                layer_index,
                LagunaLayerTensorRole::AttentionQueryNormalization,
            ),
            ones(runtime, &[2]),
        );
        tensors.insert(
            layer_weight_id(
                layer_index,
                LagunaLayerTensorRole::AttentionKeyNormalization,
            ),
            ones(runtime, &[2]),
        );
        if layer_index == 0 {
            tensors.insert(
                layer_weight_id(
                    layer_index,
                    LagunaLayerTensorRole::Attention(LagunaAttentionProjection::Gate),
                ),
                ones(runtime, &[4, 8]),
            );
        }
        tensors.insert(
            layer_weight_id(
                layer_index,
                LagunaLayerTensorRole::DenseFeedForward(LagunaExpertProjection::Gate),
            ),
            ones(runtime, &[16, 8]),
        );
        tensors.insert(
            layer_weight_id(
                layer_index,
                LagunaLayerTensorRole::DenseFeedForward(LagunaExpertProjection::Up),
            ),
            ones(runtime, &[16, 8]),
        );
        tensors.insert(
            layer_weight_id(
                layer_index,
                LagunaLayerTensorRole::DenseFeedForward(LagunaExpertProjection::Down),
            ),
            ones(runtime, &[8, 16]),
        );
    }
    LagunaNativeWeights::bind(runtime, tensors, contract).expect("tiny native weights should bind")
}

fn ones(runtime: &MlxRuntime, shape: &[i32]) -> MlxArray {
    let element_count = shape.iter().product::<i32>() as usize;
    runtime
        .array_from_f32(&vec![0.05; element_count], shape)
        .expect("a dense ones-scaled tensor should be valid")
}

fn weight_id(role: LagunaGlobalTensorRole) -> LagunaTensorId {
    LagunaTensorId::Global {
        role,
        component: LagunaTensorComponent::Weight,
    }
}

fn layer_weight_id(layer_index: usize, role: LagunaLayerTensorRole) -> LagunaTensorId {
    LagunaTensorId::Layer {
        layer_index,
        role,
        component: LagunaTensorComponent::Weight,
    }
}
