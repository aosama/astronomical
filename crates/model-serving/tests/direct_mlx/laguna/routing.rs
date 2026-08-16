use astronomical_model_serving::{
    LagunaFeedForwardDescriptor, LagunaTargetNormalizer, route_laguna_native_experts,
    select_laguna_router_experts,
};
use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};
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
        .expect("Laguna router test memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize")
}

fn tiny_moe_descriptor(
    experts_per_token: u32,
    normalizes: bool,
) -> astronomical_model_serving::LagunaMoeDescriptor {
    let config = json!({
        "architectures": ["LagunaForCausalLM"],
        "model_type": "laguna",
        "vocab_size": 8,
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 2,
        "max_position_embeddings": 32,
        "rms_norm_eps": 0.00001,
        "tie_word_embeddings": false,
        "torch_dtype": "float32",
        "mlp_layer_types": ["sparse"],
        "num_experts": 4,
        "num_experts_per_tok": experts_per_token,
        "moe_intermediate_size": 8,
        "shared_expert_intermediate_size": 8,
        "norm_topk_prob": normalizes,
        "moe_routed_scaling_factor": 2.5,
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 1.0
        }
    });
    let contract =
        LagunaTargetNormalizer::normalize(&serde_json::to_vec(&config).expect("config bytes"))
            .expect("tiny Mixture-of-Experts contract should normalize");
    match contract.layers()[0].feed_forward() {
        LagunaFeedForwardDescriptor::Moe(moe) => *moe,
        LagunaFeedForwardDescriptor::Dense(_) => {
            panic!("the tiny contract should expose a sparse descriptor")
        }
    }
}

#[tokio::test]
async fn should_match_cpu_router_selection_for_unique_scores_with_bias_and_normalization() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = test_runtime();
    let moe = tiny_moe_descriptor(2, true);
    let logits = [0.0_f32, 2.0, -2.0, 0.5];
    let bias = [2.0_f32, 0.0, 0.0, 0.0];
    let cpu = select_laguna_router_experts(&logits, 1, 4, 2, Some(&bias), 0.0, true)
        .expect("CPU selection should succeed");
    let gpu_logits = runtime
        .array_from_f32(&logits, &[1, 4])
        .expect("logits should be valid");
    let gpu_bias = runtime
        .array_from_f32(&bias, &[4])
        .expect("bias should be valid");
    let (gpu_indices, gpu_scores) =
        route_laguna_native_experts(&runtime, &gpu_logits, Some(&gpu_bias), &moe, 0.0)
            .expect("GPU selection should succeed");
    let mut gpu_index_values = gpu_indices
        .to_vec_u32()
        .expect("selected ids should evaluate");
    let mut gpu_score_values = runtime
        .astype(
            &gpu_scores,
            astronomical_runtime_integration::MlxDtype::Float32,
        )
        .expect("scores should cast")
        .to_vec_f32()
        .expect("selected scores should evaluate");
    let mut cpu_pairs = cpu
        .expert_indices()
        .iter()
        .copied()
        .zip(cpu.original_scores().iter().copied())
        .collect::<Vec<_>>();
    let mut gpu_pairs = gpu_index_values
        .drain(..)
        .zip(gpu_score_values.drain(..))
        .collect::<Vec<_>>();
    cpu_pairs.sort_by_key(|(expert_index, _)| *expert_index);
    gpu_pairs.sort_by_key(|(expert_index, _)| *expert_index);
    assert_eq!(
        gpu_pairs
            .iter()
            .map(|(expert_index, _)| *expert_index)
            .collect::<Vec<_>>(),
        cpu_pairs
            .iter()
            .map(|(expert_index, _)| *expert_index)
            .collect::<Vec<_>>()
    );
    for ((_, gpu_score), (_, cpu_score)) in gpu_pairs.iter().zip(&cpu_pairs) {
        assert!(
            (gpu_score - cpu_score).abs() < 1e-5,
            "expected GPU score {gpu_score} to match CPU {cpu_score}"
        );
    }
}

#[tokio::test]
async fn should_change_gathered_scores_when_positive_softcap_is_enabled() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = test_runtime();
    let moe = tiny_moe_descriptor(1, false);
    let logits = [30.0_f32, 0.0];
    let gpu_logits = runtime
        .array_from_f32(&logits, &[1, 2])
        .expect("logits should be valid");
    let (_uncapped_indices, uncapped_scores) =
        route_laguna_native_experts(&runtime, &gpu_logits, None, &moe, 0.0)
            .expect("disabled softcap should route");
    let (_capped_indices, capped_scores) =
        route_laguna_native_experts(&runtime, &gpu_logits, None, &moe, 2.0)
            .expect("positive softcap should route");
    let uncapped = runtime
        .astype(
            &uncapped_scores,
            astronomical_runtime_integration::MlxDtype::Float32,
        )
        .and_then(|scores| scores.to_vec_f32())
        .expect("uncapped scores should evaluate");
    let capped = runtime
        .astype(
            &capped_scores,
            astronomical_runtime_integration::MlxDtype::Float32,
        )
        .and_then(|scores| scores.to_vec_f32())
        .expect("capped scores should evaluate");
    assert!(uncapped[0] > capped[0]);
}
