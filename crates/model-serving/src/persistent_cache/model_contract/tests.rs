use crate::decoder_cache::{
    DecoderCacheLayerLayout, DecoderCacheLayout, DecoderCacheTensorDtype, DecoderCacheTensorLayout,
};
use crate::persistent_cache::model_contract_error::PersistentPromptCacheModelContractError;
use crate::qwen3_5_core::configuration::Qwen3_5Config;
use crate::qwen3_5_core::decoder::cache_layout::{
    Qwen3_5DecoderLayerCacheDtypes, qwen3_5_decoder_cache_layout,
};
use crate::qwen3_5_core::frozen_config_test_support::{
    frozen_ornith_1_0_config, persistent_visual_embedding_model_contract,
};
use crate::qwen3_5_streaming::{
    ORNITH_1_0_35B_OPTIQ_4BIT_MODEL_ID, ORNITH_1_0_35B_OPTIQ_4BIT_REVISION,
};

use super::PersistentPromptCacheModelContract;

const TEST_MLX_MEMORY_CEILING_BYTES: u64 = 20_000_000_000;
const TEST_SSD_QUOTA_BYTES: u64 = 50_000_000_000;

/// Resolves the frozen Ornith 1.0 storage contract (and the frozen config it
/// was resolved from, for the position-count invariant) so each facet test
/// below holds only the facts it verifies.
fn frozen_ornith_storage_contract() -> (PersistentPromptCacheModelContract, Qwen3_5Config) {
    let ornith_config = frozen_ornith_1_0_config();
    let decoder_layer_cache_dtypes = bfloat16_decoder_layer_cache_dtypes(&ornith_config);
    let storage_contract = PersistentPromptCacheModelContract::resolve(
        ORNITH_1_0_35B_OPTIQ_4BIT_MODEL_ID.to_owned(),
        ORNITH_1_0_35B_OPTIQ_4BIT_REVISION.to_owned(),
        qwen3_5_decoder_cache_layout(&ornith_config, 256, &decoder_layer_cache_dtypes)
            .expect("the frozen Ornith 1.0 configuration should build a decoder-cache layout"),
        ornith_config.maximum_position_count() as usize,
        TEST_MLX_MEMORY_CEILING_BYTES,
        TEST_SSD_QUOTA_BYTES,
        None,
        4,
    )
    .expect("the frozen model should resolve an SSD storage contract");
    (storage_contract, ornith_config)
}

#[test]
fn should_report_frozen_model_identity_and_layer_geometry() {
    let (storage_contract, _ornith_config) = frozen_ornith_storage_contract();
    let persistent_visual_embedding_model_contract = persistent_visual_embedding_model_contract();

    assert_eq!(
        storage_contract.model_id(),
        ORNITH_1_0_35B_OPTIQ_4BIT_MODEL_ID
    );
    assert_eq!(
        storage_contract.model_revision(),
        ORNITH_1_0_35B_OPTIQ_4BIT_REVISION
    );
    assert_eq!(storage_contract.decoder_cache_layout().layer_count(), 40);
    assert_eq!(
        storage_contract
            .decoder_cache_layout()
            .sequence_tensor_count(),
        20
    );
    assert_eq!(
        storage_contract
            .decoder_cache_layout()
            .boundary_tensor_count(),
        60
    );
    assert_eq!(
        persistent_visual_embedding_model_contract.maximum_visual_embedding_token_count(),
        16_384
    );
}

#[test]
fn should_report_frozen_tensor_shapes_for_sequence_boundary_and_visual_state() {
    let (storage_contract, _ornith_config) = frozen_ornith_storage_contract();
    let persistent_visual_embedding_model_contract = persistent_visual_embedding_model_contract();

    assert_eq!(
        storage_contract
            .decoder_cache_layout()
            .sequence_tensor_layouts()
            .first()
            .expect("the frozen layout should contain sequence state")
            .tensor_layout()
            .dimensions(),
        &[1, 2, 0, 256]
    );
    assert_eq!(
        storage_contract
            .decoder_cache_layout()
            .boundary_tensor_layouts()
            .first()
            .expect("the frozen layout should contain boundary state")
            .tensor_layout()
            .dimensions(),
        &[1, 3, 8_192]
    );
    assert_eq!(
        persistent_visual_embedding_model_contract.visual_embedding_shape(1_560),
        [1_560, 2_048]
    );
}

#[test]
fn should_preserve_frozen_dtype_composition_across_the_cache_layout() {
    let (storage_contract, _ornith_config) = frozen_ornith_storage_contract();
    let decoder_cache_layout = storage_contract.decoder_cache_layout();

    assert!(
        decoder_cache_layout
            .sequence_tensor_layouts()
            .iter()
            .all(|tensor_layout| tensor_layout.tensor_layout().dtype()
                == DecoderCacheTensorDtype::BFloat16)
    );
    let boundary_tensor_layouts = decoder_cache_layout.boundary_tensor_layouts();
    let mut bfloat16_boundary_count = 0_usize;
    let mut float32_boundary_count = 0_usize;
    for tensor_layout in boundary_tensor_layouts {
        match tensor_layout.tensor_layout().dtype() {
            DecoderCacheTensorDtype::BFloat16 => bfloat16_boundary_count += 1,
            DecoderCacheTensorDtype::Float32 => float32_boundary_count += 1,
            // The frozen layout pins only these two dtypes for boundary state;
            // any other variant still fails the exact counts asserted below.
            _ => {}
        }
    }
    assert_eq!(bfloat16_boundary_count, 30);
    assert_eq!(float32_boundary_count, 30);
}

#[test]
fn should_account_frozen_payload_bytes_per_block_within_the_position_budget() {
    let (storage_contract, ornith_config) = frozen_ornith_storage_contract();

    let block_token_count = storage_contract.block_token_count();
    assert!(block_token_count.is_multiple_of(256));
    assert!(block_token_count <= ornith_config.maximum_position_count() as usize);
    assert!(
        storage_contract.sequence_state_payload_bytes_per_block()
            >= storage_contract.boundary_state_payload_bytes()
    );
    assert_eq!(
        storage_contract.sequence_state_payload_bytes_per_block(),
        block_token_count * 20_480
    );
    assert_ne!(storage_contract.storage_contract_fingerprint(), [0_u8; 32]);
}

#[test]
fn should_preserve_mixed_execution_dtypes_in_persistent_prompt_cache_geometry() {
    let ornith_config = frozen_ornith_1_0_config();
    let mut decoder_layer_cache_dtypes = bfloat16_decoder_layer_cache_dtypes(&ornith_config);
    decoder_layer_cache_dtypes[0] = Qwen3_5DecoderLayerCacheDtypes::LinearAttention {
        convolution: DecoderCacheTensorDtype::Float32,
    };
    decoder_layer_cache_dtypes[3] = Qwen3_5DecoderLayerCacheDtypes::FullAttention {
        keys: DecoderCacheTensorDtype::Float32,
        values: DecoderCacheTensorDtype::Float32,
    };

    let decoder_cache_layout =
        qwen3_5_decoder_cache_layout(&ornith_config, 256, &decoder_layer_cache_dtypes)
            .expect("mixed execution dtypes should build an exact decoder-cache layout");

    let first_boundary_tensor = decoder_cache_layout
        .boundary_tensor_layouts()
        .into_iter()
        .find(|persisted_tensor_layout| persisted_tensor_layout.decoder_layer_index() == 0)
        .expect("the first linear-attention layer should have boundary state");
    assert_eq!(
        first_boundary_tensor.tensor_layout().dtype(),
        DecoderCacheTensorDtype::Float32
    );
    let first_full_attention_tensors = decoder_cache_layout
        .sequence_tensor_layouts()
        .into_iter()
        .filter(|persisted_tensor_layout| persisted_tensor_layout.decoder_layer_index() == 3)
        .collect::<Vec<_>>();
    assert_eq!(first_full_attention_tensors.len(), 2);
    assert!(
        first_full_attention_tensors
            .iter()
            .all(|persisted_tensor_layout| {
                persisted_tensor_layout.tensor_layout().dtype() == DecoderCacheTensorDtype::Float32
            })
    );
}

#[test]
fn should_reject_decoder_cache_execution_dtypes_with_a_missing_layer() {
    let ornith_config = frozen_ornith_1_0_config();
    let mut decoder_layer_cache_dtypes = bfloat16_decoder_layer_cache_dtypes(&ornith_config);
    decoder_layer_cache_dtypes.pop();

    let rejection = qwen3_5_decoder_cache_layout(&ornith_config, 256, &decoder_layer_cache_dtypes);

    assert!(matches!(
        rejection,
        Err(
            crate::decoder_cache::DecoderCacheLayoutError::ExecutionDtypeLayerCountMismatch {
                expected_layer_count: 40,
                actual_layer_count: 39,
            }
        )
    ));
}

#[test]
fn should_reject_decoder_cache_execution_dtypes_for_the_wrong_attention_family() {
    let ornith_config = frozen_ornith_1_0_config();
    let mut decoder_layer_cache_dtypes = bfloat16_decoder_layer_cache_dtypes(&ornith_config);
    decoder_layer_cache_dtypes[0] = Qwen3_5DecoderLayerCacheDtypes::FullAttention {
        keys: DecoderCacheTensorDtype::BFloat16,
        values: DecoderCacheTensorDtype::BFloat16,
    };

    let rejection = qwen3_5_decoder_cache_layout(&ornith_config, 256, &decoder_layer_cache_dtypes);

    assert!(matches!(
        rejection,
        Err(
            crate::decoder_cache::DecoderCacheLayoutError::ExecutionDtypeLayerFamilyMismatch {
                layer_index: 0,
            }
        )
    ));
}

fn bfloat16_decoder_layer_cache_dtypes(
    ornith_config: &Qwen3_5Config,
) -> Vec<Qwen3_5DecoderLayerCacheDtypes> {
    (0..ornith_config.layer_count() as usize)
        .map(|decoder_layer_index| {
            if ornith_config.decoder_layer_is_full_attention(decoder_layer_index) {
                Qwen3_5DecoderLayerCacheDtypes::FullAttention {
                    keys: DecoderCacheTensorDtype::BFloat16,
                    values: DecoderCacheTensorDtype::BFloat16,
                }
            } else {
                Qwen3_5DecoderLayerCacheDtypes::LinearAttention {
                    convolution: DecoderCacheTensorDtype::BFloat16,
                }
            }
        })
        .collect()
}

#[test]
fn should_derive_different_block_sizes_from_sequence_and_boundary_geometry() {
    let small_boundary_contract = resolve_synthetic_contract(
        "small-boundary",
        synthetic_hybrid_layout(8),
        128,
        1_000_000,
        1_000_000,
    );
    let large_boundary_contract = resolve_synthetic_contract(
        "large-boundary",
        synthetic_hybrid_layout(64),
        128,
        1_000_000,
        1_000_000,
    );

    assert_eq!(small_boundary_contract.block_token_count(), 4);
    assert_eq!(large_boundary_contract.block_token_count(), 32);
    assert_ne!(
        small_boundary_contract.storage_contract_fingerprint(),
        large_boundary_contract.storage_contract_fingerprint()
    );
}

#[test]
fn should_resolve_sequence_only_and_boundary_only_storage_contracts() {
    let sequence_only_layout =
        DecoderCacheLayout::new(vec![DecoderCacheLayerLayout::append_only_attention(
            DecoderCacheTensorLayout::sequence(
                "attention.keys",
                DecoderCacheTensorDtype::Float16,
                vec![1, 0, 4],
                1,
            ),
            DecoderCacheTensorLayout::sequence(
                "attention.values",
                DecoderCacheTensorDtype::Float16,
                vec![1, 0, 4],
                1,
            ),
            16,
        )])
        .expect("the sequence-only layout should be valid");
    let boundary_only_layout =
        DecoderCacheLayout::new(vec![DecoderCacheLayerLayout::recurrent_tensor(
            DecoderCacheTensorLayout::fixed(
                "recurrent.state",
                DecoderCacheTensorDtype::Float32,
                vec![25],
            ),
        )])
        .expect("the boundary-only layout should be valid");

    let sequence_only_contract = resolve_synthetic_contract(
        "sequence-only",
        sequence_only_layout,
        128,
        1_000_000,
        1_000_000,
    );
    let boundary_only_contract = resolve_synthetic_contract(
        "boundary-only",
        boundary_only_layout,
        100,
        1_000_000,
        10_000,
    );

    assert_eq!(sequence_only_contract.block_token_count(), 16);
    assert!(sequence_only_contract.has_sequence_state());
    assert!(!sequence_only_contract.has_boundary_state());
    assert!(boundary_only_contract.block_token_count() <= 100);
    assert!(!boundary_only_contract.has_sequence_state());
    assert!(boundary_only_contract.has_boundary_state());
    for storage_contract in [&sequence_only_contract, &boundary_only_contract] {
        let maximum_committed_block_count = storage_contract
            .maximum_context_token_count()
            .div_ceil(storage_contract.block_token_count());
        assert!(
            storage_contract
                .maximum_committed_block_bytes()
                .saturating_mul(maximum_committed_block_count as u64)
                <= if storage_contract.has_sequence_state() {
                    1_000_000
                } else {
                    10_000
                },
            "the complete active chain must fit its exact committed-byte quota"
        );
    }
}

#[test]
fn should_reject_a_contract_when_one_exact_capture_exceeds_a_budget() {
    let rejected_contract = PersistentPromptCacheModelContract::resolve(
        "fictional-model".to_owned(),
        "revision".to_owned(),
        synthetic_hybrid_layout(64),
        128,
        100,
        1_000_000,
        None,
        4,
    );

    assert!(rejected_contract.is_err());
}

#[test]
fn should_apply_configured_prompt_cache_block_and_common_prefix_boundaries() {
    let configured_contract = PersistentPromptCacheModelContract::resolve(
        "fictional-model".to_owned(),
        "revision".to_owned(),
        synthetic_hybrid_layout(8),
        128,
        1_000_000,
        1_000_000,
        Some(16),
        6,
    )
    .expect("aligned configured cache boundaries should resolve");

    assert_eq!(configured_contract.block_token_count(), 16);
    assert_eq!(
        configured_contract.common_prefix_checkpoint_stride_blocks(),
        6
    );
}

#[test]
fn should_reject_a_configured_prompt_cache_block_that_breaks_model_alignment() {
    let rejected_contract = PersistentPromptCacheModelContract::resolve(
        "fictional-model".to_owned(),
        "revision".to_owned(),
        synthetic_hybrid_layout(8),
        128,
        1_000_000,
        1_000_000,
        Some(6),
        4,
    );

    assert!(rejected_contract.is_err());
}

#[test]
fn should_reject_zero_common_prefix_checkpoint_stride_at_the_storage_boundary() {
    let rejected_contract = PersistentPromptCacheModelContract::resolve(
        "fictional-model".to_owned(),
        "revision".to_owned(),
        synthetic_hybrid_layout(8),
        128,
        1_000_000,
        1_000_000,
        Some(16),
        0,
    );

    assert!(matches!(
        rejected_contract,
        Err(PersistentPromptCacheModelContractError::ZeroCommonPrefixCheckpointStrideBlocks)
    ));
}

#[test]
fn should_reject_an_explicit_block_length_instead_of_silently_resizing_it_for_quota() {
    let rejected_contract = PersistentPromptCacheModelContract::resolve(
        "fictional-model".to_owned(),
        "revision".to_owned(),
        synthetic_hybrid_layout(8),
        128,
        1_000_000,
        1,
        Some(16),
        4,
    );

    assert!(matches!(
        rejected_contract,
        Err(
            PersistentPromptCacheModelContractError::ConfiguredBlockChainExceedsSsdQuota {
                configured_block_tokens: 16,
                ..
            }
        )
    ));
}

fn resolve_synthetic_contract(
    model_id: &str,
    decoder_cache_layout: DecoderCacheLayout,
    maximum_context_token_count: usize,
    effective_mlx_memory_ceiling_bytes: u64,
    global_ssd_quota_bytes: u64,
) -> PersistentPromptCacheModelContract {
    PersistentPromptCacheModelContract::resolve(
        model_id.to_owned(),
        "revision".to_owned(),
        decoder_cache_layout,
        maximum_context_token_count,
        effective_mlx_memory_ceiling_bytes,
        global_ssd_quota_bytes,
        None,
        4,
    )
    .expect("the synthetic model should resolve an SSD storage contract")
}

fn synthetic_hybrid_layout(boundary_element_count: usize) -> DecoderCacheLayout {
    DecoderCacheLayout::new(vec![DecoderCacheLayerLayout::composite(vec![
        DecoderCacheLayerLayout::append_only_attention(
            DecoderCacheTensorLayout::sequence(
                "attention.keys",
                DecoderCacheTensorDtype::Float16,
                vec![1, 0, 2],
                1,
            ),
            DecoderCacheTensorLayout::sequence(
                "attention.values",
                DecoderCacheTensorDtype::Float16,
                vec![1, 0, 2],
                1,
            ),
            4,
        ),
        DecoderCacheLayerLayout::recurrent_tensor(DecoderCacheTensorLayout::fixed(
            "recurrent.state",
            DecoderCacheTensorDtype::Float32,
            vec![boundary_element_count],
        )),
    ])])
    .expect("the synthetic hybrid layout should be valid")
}
