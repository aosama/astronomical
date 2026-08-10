use astronomical_model_serving::{
    DecoderCacheLayerLayout, DecoderCacheLayout, DecoderCacheTensorDtype, DecoderCacheTensorLayout,
    ORNITH_1_0_35B_OPTIQ_4BIT_MODEL_ID, ORNITH_1_0_35B_OPTIQ_4BIT_REVISION,
    PersistentPromptCacheModelContract, qwen3_5_decoder_cache_layout,
};

const TEST_MLX_MEMORY_CEILING_BYTES: u64 = 20_000_000_000;
const TEST_SSD_QUOTA_BYTES: u64 = 50_000_000_000;

use crate::common::qwen3_5_moe::{
    certified_ornith_config, persistent_visual_embedding_model_contract,
};

#[test]
fn should_derive_persistent_prompt_cache_tensor_shapes_from_certified_model_metadata() {
    let ornith_config = certified_ornith_config();

    let persistent_prompt_cache_model_contract = PersistentPromptCacheModelContract::resolve(
        ORNITH_1_0_35B_OPTIQ_4BIT_MODEL_ID.to_owned(),
        ORNITH_1_0_35B_OPTIQ_4BIT_REVISION.to_owned(),
        qwen3_5_decoder_cache_layout(&ornith_config, 256)
            .expect("the certified Ornith configuration should build a decoder-cache layout"),
        ornith_config.maximum_position_count() as usize,
        TEST_MLX_MEMORY_CEILING_BYTES,
        TEST_SSD_QUOTA_BYTES,
        None,
        4,
    )
    .expect("the certified model should resolve an SSD storage contract");
    let persistent_visual_embedding_model_contract = persistent_visual_embedding_model_contract();

    assert_eq!(
        persistent_prompt_cache_model_contract.model_id(),
        ORNITH_1_0_35B_OPTIQ_4BIT_MODEL_ID
    );
    assert_eq!(
        persistent_prompt_cache_model_contract.model_revision(),
        ORNITH_1_0_35B_OPTIQ_4BIT_REVISION
    );
    assert_eq!(
        persistent_prompt_cache_model_contract
            .decoder_cache_layout()
            .layer_count(),
        40
    );
    assert_eq!(
        persistent_prompt_cache_model_contract
            .decoder_cache_layout()
            .sequence_tensor_layouts()
            .first()
            .expect("the certified layout should contain sequence state")
            .tensor_layout()
            .dimensions(),
        &[1, 2, 0, 256]
    );
    assert_eq!(
        persistent_prompt_cache_model_contract
            .decoder_cache_layout()
            .boundary_tensor_layouts()
            .first()
            .expect("the certified layout should contain boundary state")
            .tensor_layout()
            .dimensions(),
        &[1, 3, 8_192]
    );
    assert_eq!(
        persistent_visual_embedding_model_contract.visual_embedding_shape(1_560),
        [1_560, 2_048]
    );
    assert_eq!(
        persistent_visual_embedding_model_contract.maximum_visual_embedding_token_count(),
        16_384
    );
    assert_eq!(
        persistent_prompt_cache_model_contract
            .decoder_cache_layout()
            .sequence_tensor_count(),
        20
    );
    assert_eq!(
        persistent_prompt_cache_model_contract
            .decoder_cache_layout()
            .boundary_tensor_count(),
        60
    );
    let decoder_cache_layout = persistent_prompt_cache_model_contract.decoder_cache_layout();
    assert!(
        decoder_cache_layout
            .sequence_tensor_layouts()
            .iter()
            .all(|tensor_layout| tensor_layout.tensor_layout().dtype()
                == DecoderCacheTensorDtype::BFloat16)
    );
    let boundary_tensor_layouts = decoder_cache_layout.boundary_tensor_layouts();
    assert_eq!(
        boundary_tensor_layouts
            .iter()
            .filter(|tensor_layout| {
                tensor_layout.tensor_layout().dtype() == DecoderCacheTensorDtype::BFloat16
            })
            .count(),
        30
    );
    assert_eq!(
        boundary_tensor_layouts
            .iter()
            .filter(|tensor_layout| {
                tensor_layout.tensor_layout().dtype() == DecoderCacheTensorDtype::Float32
            })
            .count(),
        30
    );
    let block_token_count = persistent_prompt_cache_model_contract.block_token_count();
    assert!(block_token_count.is_multiple_of(256));
    assert!(block_token_count <= ornith_config.maximum_position_count() as usize);
    assert!(
        persistent_prompt_cache_model_contract.sequence_state_payload_bytes_per_block()
            >= persistent_prompt_cache_model_contract.boundary_state_payload_bytes()
    );
    assert_eq!(
        persistent_prompt_cache_model_contract.sequence_state_payload_bytes_per_block(),
        block_token_count * 20_480
    );
    assert_ne!(
        persistent_prompt_cache_model_contract.storage_contract_fingerprint(),
        [0_u8; 32]
    );
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
        Err(astronomical_model_serving::PersistentPromptCacheModelContractError::ZeroCommonPrefixCheckpointStrideBlocks)
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
        Err(astronomical_model_serving::PersistentPromptCacheModelContractError::ConfiguredBlockChainExceedsSsdQuota {
            configured_block_tokens: 16,
            ..
        })
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
