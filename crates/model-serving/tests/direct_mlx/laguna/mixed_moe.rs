use std::collections::HashMap;

use astronomical_model_serving::{
    ExpertMemoryMode, LagunaAttentionProjection, LagunaDecoderState, LagunaExpertProjection,
    LagunaGlobalTensorRole, LagunaLayerTensorRole, LagunaModel, LagunaNativeWeights,
    LagunaTargetNormalizer, LagunaTensorComponent, LagunaTensorId, PerformanceAttribution,
    PerformanceOperation,
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
        .expect("Laguna Mixture-of-Experts test memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize")
}

fn filled(runtime: &MlxRuntime, shape: &[i32], fill: f32) -> MlxArray {
    let element_count = shape.iter().product::<i32>() as usize;
    runtime
        .array_from_f32(&vec![fill; element_count], shape)
        .expect("a filled tensor should be valid")
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

fn dense_then_sparse_contract() -> astronomical_model_serving::LagunaTargetContract {
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
        "mlp_layer_types": ["dense", "sparse"],
        "gating_types": ["none", "none"],
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 8,
        "shared_expert_intermediate_size": 8,
        "norm_topk_prob": true,
        "moe_routed_scaling_factor": 2.5,
        "moe_apply_router_weight_on_input": false,
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 1.0
        }
    });
    LagunaTargetNormalizer::normalize(&serde_json::to_vec(&config).expect("config bytes"))
        .expect("dense-then-sparse contract should normalize")
}

fn bind_dense_then_sparse_weights(
    runtime: &MlxRuntime,
    contract: &astronomical_model_serving::LagunaTargetContract,
    include_router: bool,
) -> Result<LagunaNativeWeights, astronomical_model_serving::LagunaExecutionError> {
    let mut tensors = HashMap::new();
    tensors.insert(
        weight_id(LagunaGlobalTensorRole::TokenEmbedding),
        filled(runtime, &[8, 8], 0.05),
    );
    tensors.insert(
        weight_id(LagunaGlobalTensorRole::FinalNormalization),
        filled(runtime, &[8], 1.0),
    );
    tensors.insert(
        weight_id(LagunaGlobalTensorRole::OutputHead),
        filled(runtime, &[8, 8], 0.05),
    );
    for layer_index in 0..2 {
        tensors.insert(
            layer_weight_id(layer_index, LagunaLayerTensorRole::InputNormalization),
            filled(runtime, &[8], 1.0),
        );
        tensors.insert(
            layer_weight_id(
                layer_index,
                LagunaLayerTensorRole::PostAttentionNormalization,
            ),
            filled(runtime, &[8], 1.0),
        );
        tensors.insert(
            layer_weight_id(
                layer_index,
                LagunaLayerTensorRole::Attention(LagunaAttentionProjection::Query),
            ),
            filled(runtime, &[8, 8], 0.05),
        );
        tensors.insert(
            layer_weight_id(
                layer_index,
                LagunaLayerTensorRole::Attention(LagunaAttentionProjection::Key),
            ),
            filled(runtime, &[4, 8], 0.05),
        );
        tensors.insert(
            layer_weight_id(
                layer_index,
                LagunaLayerTensorRole::Attention(LagunaAttentionProjection::Value),
            ),
            filled(runtime, &[4, 8], 0.05),
        );
        tensors.insert(
            layer_weight_id(
                layer_index,
                LagunaLayerTensorRole::Attention(LagunaAttentionProjection::Output),
            ),
            filled(runtime, &[8, 8], 0.05),
        );
        tensors.insert(
            layer_weight_id(
                layer_index,
                LagunaLayerTensorRole::AttentionQueryNormalization,
            ),
            filled(runtime, &[2], 1.0),
        );
        tensors.insert(
            layer_weight_id(
                layer_index,
                LagunaLayerTensorRole::AttentionKeyNormalization,
            ),
            filled(runtime, &[2], 1.0),
        );
    }
    for projection in [
        LagunaExpertProjection::Gate,
        LagunaExpertProjection::Up,
        LagunaExpertProjection::Down,
    ] {
        let shape = if matches!(projection, LagunaExpertProjection::Down) {
            [8, 16]
        } else {
            [16, 8]
        };
        tensors.insert(
            layer_weight_id(0, LagunaLayerTensorRole::DenseFeedForward(projection)),
            filled(runtime, &shape, 0.05),
        );
    }
    if include_router {
        tensors.insert(
            layer_weight_id(1, LagunaLayerTensorRole::Router),
            filled(runtime, &[4, 8], 0.1),
        );
        tensors.insert(
            layer_weight_id(1, LagunaLayerTensorRole::RouterCorrectionBias),
            runtime
                .array_from_f32(&[0.4, 0.0, 0.0, 0.0], &[4])
                .expect("correction bias should be valid"),
        );
        tensors.insert(
            layer_weight_id(
                1,
                LagunaLayerTensorRole::RoutedExpert(LagunaExpertProjection::Gate),
            ),
            filled(runtime, &[4, 8, 8], 0.05),
        );
        tensors.insert(
            layer_weight_id(
                1,
                LagunaLayerTensorRole::RoutedExpert(LagunaExpertProjection::Up),
            ),
            filled(runtime, &[4, 8, 8], 0.05),
        );
        tensors.insert(
            layer_weight_id(
                1,
                LagunaLayerTensorRole::RoutedExpert(LagunaExpertProjection::Down),
            ),
            filled(runtime, &[4, 8, 8], 0.05),
        );
        for projection in [
            LagunaExpertProjection::Gate,
            LagunaExpertProjection::Up,
            LagunaExpertProjection::Down,
        ] {
            let shape = if matches!(projection, LagunaExpertProjection::Down) {
                [8, 8]
            } else {
                [8, 8]
            };
            tensors.insert(
                layer_weight_id(1, LagunaLayerTensorRole::SharedExpert(projection)),
                filled(runtime, &shape, 0.05),
            );
        }
    }
    LagunaNativeWeights::bind(runtime, tensors, contract)
}

#[tokio::test]
async fn should_run_dense_then_sparse_prefill_and_decode_with_attribution() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = test_runtime();
    let contract = dense_then_sparse_contract();
    let weights = bind_dense_then_sparse_weights(&runtime, &contract, true)
        .expect("dense-then-sparse weights should bind");
    let model = LagunaModel::new(
        contract,
        weights,
        crate::common::test_worker_kernel_capabilities(&runtime),
    )
    .expect("the mixed model should construct");
    let mut decoder_state =
        LagunaDecoderState::empty(model.contract()).expect("decoder state should allocate");
    let mut performance_attribution = PerformanceAttribution::enabled();
    let prompt_tokens = runtime
        .array_from_u32(&[1, 2], &[2])
        .expect("prompt token ids should be valid");
    let prompt_logits = model
        .forward(
            &runtime,
            &prompt_tokens,
            &mut decoder_state,
            &mut performance_attribution,
        )
        .expect("dense-then-sparse prefill should execute");
    assert_eq!(prompt_logits.shape(), vec![1, 1, 8]);
    assert_eq!(decoder_state.absolute_position(0), Some(2));
    assert_eq!(decoder_state.committed_token_count(1), Some(2));

    let decode_tokens = runtime
        .array_from_u32(&[3], &[1])
        .expect("decode token ids should be valid");
    let decode_logits = model
        .forward(
            &runtime,
            &decode_tokens,
            &mut decoder_state,
            &mut performance_attribution,
        )
        .expect("dense-then-sparse decode should execute");
    assert_eq!(decode_logits.shape(), vec![1, 1, 8]);
    assert!(
        performance_attribution
            .operation_measurement(PerformanceOperation::RouterScoreSelection)
            .is_some()
    );
    assert!(
        performance_attribution
            .operation_measurement(PerformanceOperation::SharedExpertExecution)
            .is_some()
    );
    assert!(
        performance_attribution
            .operation_measurement(PerformanceOperation::ResidentMoeGraphConstruction)
            .is_some()
    );
    assert_eq!(model.expert_memory_mode(), ExpertMemoryMode::Resident);
    let telemetry = model.expert_residency_telemetry();
    assert_eq!(telemetry.total_layer_count, 1);
    assert!(telemetry.resident_expert_count > 0);
    assert!(telemetry.resident_expert_payload_bytes > 0);
    assert_eq!(
        model
            .expert_weight_memory_cache_statistics()
            .disk_page_load_count,
        0
    );
}

#[tokio::test]
async fn should_reject_a_missing_router_weight_during_binding() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = test_runtime();
    let contract = dense_then_sparse_contract();
    let rejection = bind_dense_then_sparse_weights(&runtime, &contract, false);
    assert!(rejection.is_err());
}
