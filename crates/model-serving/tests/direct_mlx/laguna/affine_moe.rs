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

pub(crate) fn test_runtime() -> MlxRuntime {
    MlxRuntime::initialize(
        MlxMemoryLimits::new(
            DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES,
            DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
        )
        .expect("Laguna affine test memory limits should be valid"),
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

fn layer_id(
    layer_index: usize,
    role: LagunaLayerTensorRole,
    component: LagunaTensorComponent,
) -> LagunaTensorId {
    LagunaTensorId::Layer {
        layer_index,
        role,
        component,
    }
}

fn insert_affine(
    runtime: &MlxRuntime,
    tensors: &mut HashMap<LagunaTensorId, MlxArray>,
    tensor_id: LagunaTensorId,
    native_shape: &[i32],
    bits: i32,
    group_size: i32,
) {
    let native = filled(runtime, native_shape, 0.05);
    let (packed, scales, biases) = runtime
        .quantize_affine(&native, group_size, bits)
        .expect("affine quantization should succeed");
    tensors.insert(tensor_id, packed);
    tensors.insert(
        with_component(tensor_id, LagunaTensorComponent::Scales),
        scales,
    );
    tensors.insert(
        with_component(tensor_id, LagunaTensorComponent::Biases),
        biases,
    );
}

fn with_component(tensor_id: LagunaTensorId, component: LagunaTensorComponent) -> LagunaTensorId {
    match tensor_id {
        LagunaTensorId::Global { role, .. } => LagunaTensorId::Global { role, component },
        LagunaTensorId::Layer {
            layer_index, role, ..
        } => LagunaTensorId::Layer {
            layer_index,
            role,
            component,
        },
    }
}

pub(crate) fn affine_sparse_contract(
    bits: u32,
    group_size: u32,
    max_position_embeddings: u32,
) -> astronomical_model_serving::LagunaTargetContract {
    let config = json!({
        "architectures": ["LagunaForCausalLM"],
        "model_type": "laguna",
        "vocab_size": 8,
        "hidden_size": 32,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "max_position_embeddings": max_position_embeddings,
        "rms_norm_eps": 0.00001,
        "tie_word_embeddings": false,
        "torch_dtype": "float32",
        "mlp_layer_types": ["sparse"],
        "gating_types": ["none"],
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 32,
        "shared_expert_intermediate_size": 32,
        "norm_topk_prob": true,
        "moe_routed_scaling_factor": 2.5,
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 1.0
        },
        "quantization": {
            "bits": bits,
            "group_size": group_size,
            "mode": "affine"
        }
    });
    LagunaTargetNormalizer::normalize(&serde_json::to_vec(&config).expect("config bytes"))
        .expect("affine sparse contract should normalize")
}

pub(crate) fn bind_affine_sparse_model(
    runtime: &MlxRuntime,
    bits: i32,
    group_size: i32,
    include_sidecars: bool,
) -> Result<
    (
        astronomical_model_serving::LagunaTargetContract,
        LagunaNativeWeights,
    ),
    astronomical_model_serving::LagunaExecutionError,
> {
    let contract = affine_sparse_contract(bits as u32, group_size as u32, 32);
    let mut tensors = HashMap::new();
    tensors.insert(
        weight_id(LagunaGlobalTensorRole::TokenEmbedding),
        filled(runtime, &[8, 32], 0.05),
    );
    tensors.insert(
        weight_id(LagunaGlobalTensorRole::FinalNormalization),
        filled(runtime, &[32], 1.0),
    );
    insert_affine(
        runtime,
        &mut tensors,
        weight_id(LagunaGlobalTensorRole::OutputHead),
        &[8, 32],
        bits,
        group_size,
    );
    tensors.insert(
        layer_id(
            0,
            LagunaLayerTensorRole::InputNormalization,
            LagunaTensorComponent::Weight,
        ),
        filled(runtime, &[32], 1.0),
    );
    tensors.insert(
        layer_id(
            0,
            LagunaLayerTensorRole::PostAttentionNormalization,
            LagunaTensorComponent::Weight,
        ),
        filled(runtime, &[32], 1.0),
    );
    tensors.insert(
        layer_id(
            0,
            LagunaLayerTensorRole::AttentionQueryNormalization,
            LagunaTensorComponent::Weight,
        ),
        filled(runtime, &[8], 1.0),
    );
    tensors.insert(
        layer_id(
            0,
            LagunaLayerTensorRole::AttentionKeyNormalization,
            LagunaTensorComponent::Weight,
        ),
        filled(runtime, &[8], 1.0),
    );
    insert_affine(
        runtime,
        &mut tensors,
        layer_id(
            0,
            LagunaLayerTensorRole::Attention(LagunaAttentionProjection::Query),
            LagunaTensorComponent::Weight,
        ),
        &[32, 32],
        bits,
        group_size,
    );
    insert_affine(
        runtime,
        &mut tensors,
        layer_id(
            0,
            LagunaLayerTensorRole::Attention(LagunaAttentionProjection::Key),
            LagunaTensorComponent::Weight,
        ),
        &[16, 32],
        bits,
        group_size,
    );
    insert_affine(
        runtime,
        &mut tensors,
        layer_id(
            0,
            LagunaLayerTensorRole::Attention(LagunaAttentionProjection::Value),
            LagunaTensorComponent::Weight,
        ),
        &[16, 32],
        bits,
        group_size,
    );
    insert_affine(
        runtime,
        &mut tensors,
        layer_id(
            0,
            LagunaLayerTensorRole::Attention(LagunaAttentionProjection::Output),
            LagunaTensorComponent::Weight,
        ),
        &[32, 32],
        bits,
        group_size,
    );
    insert_affine(
        runtime,
        &mut tensors,
        layer_id(
            0,
            LagunaLayerTensorRole::Router,
            LagunaTensorComponent::Weight,
        ),
        &[4, 32],
        bits,
        group_size,
    );
    if include_sidecars {
        insert_affine(
            runtime,
            &mut tensors,
            layer_id(
                0,
                LagunaLayerTensorRole::RoutedExpert(LagunaExpertProjection::Gate),
                LagunaTensorComponent::Weight,
            ),
            &[4, 32, 32],
            bits,
            group_size,
        );
        insert_affine(
            runtime,
            &mut tensors,
            layer_id(
                0,
                LagunaLayerTensorRole::RoutedExpert(LagunaExpertProjection::Up),
                LagunaTensorComponent::Weight,
            ),
            &[4, 32, 32],
            bits,
            group_size,
        );
        insert_affine(
            runtime,
            &mut tensors,
            layer_id(
                0,
                LagunaLayerTensorRole::RoutedExpert(LagunaExpertProjection::Down),
                LagunaTensorComponent::Weight,
            ),
            &[4, 32, 32],
            bits,
            group_size,
        );
        insert_affine(
            runtime,
            &mut tensors,
            layer_id(
                0,
                LagunaLayerTensorRole::SharedExpert(LagunaExpertProjection::Gate),
                LagunaTensorComponent::Weight,
            ),
            &[32, 32],
            bits,
            group_size,
        );
        insert_affine(
            runtime,
            &mut tensors,
            layer_id(
                0,
                LagunaLayerTensorRole::SharedExpert(LagunaExpertProjection::Up),
                LagunaTensorComponent::Weight,
            ),
            &[32, 32],
            bits,
            group_size,
        );
        insert_affine(
            runtime,
            &mut tensors,
            layer_id(
                0,
                LagunaLayerTensorRole::SharedExpert(LagunaExpertProjection::Down),
                LagunaTensorComponent::Weight,
            ),
            &[32, 32],
            bits,
            group_size,
        );
    } else {
        tensors.insert(
            layer_id(
                0,
                LagunaLayerTensorRole::RoutedExpert(LagunaExpertProjection::Gate),
                LagunaTensorComponent::Weight,
            ),
            filled(runtime, &[4, 32, 32], 0.05),
        );
    }
    let weights = LagunaNativeWeights::bind(runtime, tensors, &contract)?;
    Ok((contract, weights))
}

#[tokio::test]
async fn should_execute_affine_sparse_prefill_for_supported_bit_widths() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = test_runtime();
    for bits in [2, 3, 4, 6, 8] {
        let (contract, weights) = bind_affine_sparse_model(&runtime, bits, 32, true)
            .unwrap_or_else(|_| panic!("{bits}-bit affine weights should bind"));
        let model = LagunaModel::new(
            contract,
            weights,
            crate::common::test_worker_kernel_capabilities(&runtime),
        )
        .expect("affine model should construct");
        let mut decoder_state =
            LagunaDecoderState::empty(model.contract()).expect("decoder state should allocate");
        let mut performance_attribution = PerformanceAttribution::enabled();
        let prompt_tokens = runtime
            .array_from_u32(&[1, 2], &[2])
            .expect("prompt token ids should be valid");
        let logits = model
            .forward(
                &runtime,
                &prompt_tokens,
                &mut decoder_state,
                &mut performance_attribution,
            )
            .unwrap_or_else(|_| panic!("{bits}-bit affine prefill should execute"));
        assert_eq!(logits.shape(), vec![1, 1, 8], "{bits}-bit logits");
        assert!(
            performance_attribution
                .operation_measurement(PerformanceOperation::GatheredExpertExecution)
                .is_some(),
            "{bits}-bit gathered execution should be attributed"
        );
    }
}

#[tokio::test]
async fn should_reject_affine_sidecars_that_are_not_paired() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = test_runtime();
    let rejection = bind_affine_sparse_model(&runtime, 4, 32, false);
    assert!(rejection.is_err());
}

#[tokio::test]
async fn should_bind_named_xs_and_s_group_sizes_on_stacked_experts() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = test_runtime();
    for (row_name, bits, group_size, hidden) in [("xs", 2, 64, 64), ("s", 2, 128, 128)] {
        let mut tensors = HashMap::new();
        let contract = LagunaTargetNormalizer::normalize(
            &serde_json::to_vec(&json!({
                "architectures": ["LagunaForCausalLM"],
                "model_type": "laguna",
                "vocab_size": 8,
                "hidden_size": hidden,
                "intermediate_size": hidden,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": hidden / 4,
                "max_position_embeddings": 32,
                "rms_norm_eps": 0.00001,
                "tie_word_embeddings": true,
                "torch_dtype": "float32",
                "mlp_layer_types": ["sparse"],
                "gating_types": ["none"],
                "num_experts": 2,
                "num_experts_per_tok": 1,
                "moe_intermediate_size": hidden,
                "shared_expert_intermediate_size": 0,
                "norm_topk_prob": true,
                "moe_routed_scaling_factor": 2.5,
                "rope_parameters": {
                    "rope_type": "default",
                    "rope_theta": 10000.0,
                    "partial_rotary_factor": 1.0
                },
                "quantization": { "bits": bits, "group_size": group_size, "mode": "affine" }
            }))
            .expect("named-row config"),
        )
        .unwrap_or_else(|_| panic!("{row_name} contract should normalize"));
        tensors.insert(
            weight_id(LagunaGlobalTensorRole::TokenEmbedding),
            filled(&runtime, &[8, hidden], 0.05),
        );
        tensors.insert(
            weight_id(LagunaGlobalTensorRole::FinalNormalization),
            filled(&runtime, &[hidden], 1.0),
        );
        tensors.insert(
            layer_id(
                0,
                LagunaLayerTensorRole::InputNormalization,
                LagunaTensorComponent::Weight,
            ),
            filled(&runtime, &[hidden], 1.0),
        );
        tensors.insert(
            layer_id(
                0,
                LagunaLayerTensorRole::PostAttentionNormalization,
                LagunaTensorComponent::Weight,
            ),
            filled(&runtime, &[hidden], 1.0),
        );
        let head_dim = hidden / 4;
        tensors.insert(
            layer_id(
                0,
                LagunaLayerTensorRole::AttentionQueryNormalization,
                LagunaTensorComponent::Weight,
            ),
            filled(&runtime, &[head_dim], 1.0),
        );
        tensors.insert(
            layer_id(
                0,
                LagunaLayerTensorRole::AttentionKeyNormalization,
                LagunaTensorComponent::Weight,
            ),
            filled(&runtime, &[head_dim], 1.0),
        );
        insert_affine(
            &runtime,
            &mut tensors,
            layer_id(
                0,
                LagunaLayerTensorRole::Attention(LagunaAttentionProjection::Query),
                LagunaTensorComponent::Weight,
            ),
            &[hidden, hidden],
            bits as i32,
            group_size as i32,
        );
        insert_affine(
            &runtime,
            &mut tensors,
            layer_id(
                0,
                LagunaLayerTensorRole::Attention(LagunaAttentionProjection::Key),
                LagunaTensorComponent::Weight,
            ),
            &[hidden / 2, hidden],
            bits as i32,
            group_size as i32,
        );
        insert_affine(
            &runtime,
            &mut tensors,
            layer_id(
                0,
                LagunaLayerTensorRole::Attention(LagunaAttentionProjection::Value),
                LagunaTensorComponent::Weight,
            ),
            &[hidden / 2, hidden],
            bits as i32,
            group_size as i32,
        );
        insert_affine(
            &runtime,
            &mut tensors,
            layer_id(
                0,
                LagunaLayerTensorRole::Attention(LagunaAttentionProjection::Output),
                LagunaTensorComponent::Weight,
            ),
            &[hidden, hidden],
            bits as i32,
            group_size as i32,
        );
        insert_affine(
            &runtime,
            &mut tensors,
            layer_id(
                0,
                LagunaLayerTensorRole::Router,
                LagunaTensorComponent::Weight,
            ),
            &[2, hidden],
            bits as i32,
            group_size as i32,
        );
        insert_affine(
            &runtime,
            &mut tensors,
            layer_id(
                0,
                LagunaLayerTensorRole::RoutedExpert(LagunaExpertProjection::Gate),
                LagunaTensorComponent::Weight,
            ),
            &[2, hidden, hidden],
            bits as i32,
            group_size as i32,
        );
        insert_affine(
            &runtime,
            &mut tensors,
            layer_id(
                0,
                LagunaLayerTensorRole::RoutedExpert(LagunaExpertProjection::Up),
                LagunaTensorComponent::Weight,
            ),
            &[2, hidden, hidden],
            bits as i32,
            group_size as i32,
        );
        insert_affine(
            &runtime,
            &mut tensors,
            layer_id(
                0,
                LagunaLayerTensorRole::RoutedExpert(LagunaExpertProjection::Down),
                LagunaTensorComponent::Weight,
            ),
            &[2, hidden, hidden],
            bits as i32,
            group_size as i32,
        );
        LagunaNativeWeights::bind(&runtime, tensors, &contract)
            .unwrap_or_else(|_| panic!("{row_name} affine stacked experts should bind"));
    }
}
