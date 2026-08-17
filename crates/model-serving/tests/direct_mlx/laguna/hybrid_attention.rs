use std::collections::HashMap;

use astronomical_model_serving::{
    LagunaAttentionProjection, LagunaDecoderState, LagunaExpertProjection, LagunaGlobalTensorRole,
    LagunaLayerTensorRole, LagunaModel, LagunaNativeWeights, LagunaTargetNormalizer,
    LagunaTensorComponent, LagunaTensorId, PerformanceAttribution, PerformanceOperation,
};
use astronomical_runtime_integration::{MlxArray, MlxMemoryLimits, MlxRuntime};
use serde_json::json;

use crate::common::{
    DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES, DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
};

fn test_runtime() -> MlxRuntime {
    MlxRuntime::initialize(
        MlxMemoryLimits::new(
            DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES,
            DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
        )
        .expect("Laguna execution test memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize")
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

fn tiny_mixed_contract() -> astronomical_model_serving::LagunaTargetContract {
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

fn bind_tiny_weights(
    runtime: &MlxRuntime,
    contract: &astronomical_model_serving::LagunaTargetContract,
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

#[tokio::test]
async fn should_grow_full_state_and_bound_sliding_state_through_prefill_and_decode() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = test_runtime();
    let contract = tiny_mixed_contract();
    let weights = bind_tiny_weights(&runtime, &contract);
    let model = LagunaModel::new(contract, weights).expect("the mixed model should construct");
    let mut decoder_state =
        LagunaDecoderState::empty(model.contract()).expect("decoder state should allocate");
    assert_eq!(decoder_state.payload_byte_count(), 0);
    let prefill_memory_projection = decoder_state
        .projected_forward_memory(model.contract(), 6)
        .expect("mixed prefill memory geometry should be exact");
    assert_eq!(prefill_memory_projection.persistent_growth_bytes(), 8_320);
    assert_eq!(
        prefill_memory_projection.sliding_temporary_workspace_bytes(),
        288
    );
    let mut performance_attribution = PerformanceAttribution::enabled();
    let prompt_tokens = runtime
        .array_from_u32(&[1, 2, 3, 4, 5, 6], &[6])
        .expect("prompt token ids should be valid");
    let prompt_logits = model
        .forward(
            &runtime,
            &prompt_tokens,
            &mut decoder_state,
            &mut performance_attribution,
        )
        .expect("mixed prefill should execute");
    assert_eq!(prompt_logits.shape(), vec![1, 1, 8]);
    assert_eq!(decoder_state.absolute_position(0), Some(6));
    assert_eq!(decoder_state.committed_token_count(1), Some(4));
    assert!(
        decoder_state.payload_byte_count() > 0,
        "a written Laguna decoder cache must report live context payload"
    );
    let decode_memory_projection = decoder_state
        .projected_forward_memory(model.contract(), 1)
        .expect("mixed decode memory geometry should be exact");
    assert_eq!(decode_memory_projection.persistent_growth_bytes(), 0);
    assert_eq!(
        decode_memory_projection.sliding_temporary_workspace_bytes(),
        0
    );

    let decode_tokens = runtime
        .array_from_u32(&[7], &[1])
        .expect("decode token ids should be valid");
    let decode_logits = model
        .forward(
            &runtime,
            &decode_tokens,
            &mut decoder_state,
            &mut performance_attribution,
        )
        .expect("mixed decode should execute");
    assert_eq!(decode_logits.shape(), vec![1, 1, 8]);
    assert_eq!(decoder_state.absolute_position(0), Some(7));
    assert_eq!(decoder_state.committed_token_count(1), Some(4));
    assert!(
        performance_attribution
            .operation_measurement(PerformanceOperation::AttentionForwardSpan)
            .is_some()
    );
}

#[tokio::test]
async fn should_reject_a_missing_canonical_weight() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = test_runtime();
    let contract = tiny_mixed_contract();
    let empty = HashMap::new();
    let rejection = LagunaNativeWeights::bind(&runtime, empty, &contract);
    assert!(rejection.is_err());
    let _ = runtime;
}
