//! Beside-source unit tests for the pure units extracted out of
//! `PersistentPromptCacheModelContract::resolve` (`model_contract.rs`):
//! input validation, block token count selection, SSD-quota block file
//! sizing, and direct publication workspace derivation.

use crate::decoder_cache::{
    DecoderCacheLayerLayout, DecoderCacheLayout, DecoderCacheTensorDtype, DecoderCacheTensorLayout,
};
use crate::persistent_cache::model_contract_error::PersistentPromptCacheModelContractError;

use super::{
    BlockFileSizingInputs, BlockTokenCountSizingInputs, block_file_geometry_within_ssd_quota,
    configured_or_derived_block_token_count,
    direct_publication_workspace_bytes_for_block_token_count, exact_state_file_bytes,
    maximum_block_manifest_file_bytes, validated_capture_budgets, validated_resolve_inputs,
};

const FIXTURE_MAXIMUM_CONTEXT_TOKEN_COUNT: usize = 128;
const FIXTURE_PERSISTENCE_ALIGNMENT_TOKEN_COUNT: usize = 4;
const FITTING_GLOBAL_SSD_QUOTA_BYTES: u64 = 1_000_000_000;

fn synthetic_hybrid_layout(boundary_element_count: usize) -> DecoderCacheLayout {
    // One composite layer: sequence payload costs 8 bytes per token and the
    // boundary snapshot costs 4 bytes per element, so fixture byte budgets
    // stay small and predictable.
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
            FIXTURE_PERSISTENCE_ALIGNMENT_TOKEN_COUNT,
        ),
        DecoderCacheLayerLayout::recurrent_tensor(DecoderCacheTensorLayout::fixed(
            "recurrent.state",
            DecoderCacheTensorDtype::Float32,
            vec![boundary_element_count],
        )),
    ])])
    .expect("the synthetic hybrid layout should be valid")
}

fn synthetic_boundary_only_layout(boundary_element_count: usize) -> DecoderCacheLayout {
    DecoderCacheLayout::new(vec![DecoderCacheLayerLayout::recurrent_tensor(
        DecoderCacheTensorLayout::fixed(
            "recurrent.state",
            DecoderCacheTensorDtype::Float32,
            vec![boundary_element_count],
        ),
    )])
    .expect("the synthetic boundary-only layout should be valid")
}

#[test]
fn should_accept_minimal_nonempty_resolve_inputs() {
    let validation =
        validated_resolve_inputs("model", "revision", &synthetic_hybrid_layout(8), 1, 1);

    assert!(matches!(validation, Ok(())));
}

#[test]
fn should_reject_an_empty_model_id() {
    let validation = validated_resolve_inputs("", "revision", &synthetic_hybrid_layout(8), 128, 4);

    assert!(matches!(
        validation,
        Err(PersistentPromptCacheModelContractError::EmptyModelId)
    ));
}

#[test]
fn should_reject_an_empty_model_revision() {
    let validation = validated_resolve_inputs("model", "", &synthetic_hybrid_layout(8), 128, 4);

    assert!(matches!(
        validation,
        Err(PersistentPromptCacheModelContractError::EmptyModelRevision)
    ));
}

#[test]
fn should_reject_a_zero_maximum_context_token_count() {
    let validation =
        validated_resolve_inputs("model", "revision", &synthetic_hybrid_layout(8), 0, 4);

    assert!(matches!(
        validation,
        Err(PersistentPromptCacheModelContractError::ZeroMaximumContextTokenCount)
    ));
}

#[test]
fn should_reject_a_zero_common_prefix_checkpoint_stride() {
    let validation =
        validated_resolve_inputs("model", "revision", &synthetic_hybrid_layout(8), 128, 0);

    assert!(matches!(
        validation,
        Err(PersistentPromptCacheModelContractError::ZeroCommonPrefixCheckpointStrideBlocks)
    ));
}

#[test]
fn should_reject_a_layout_without_any_persistent_state() {
    let empty_layout =
        DecoderCacheLayout::new(vec![]).expect("an empty layout should still construct");

    let validation = validated_resolve_inputs("model", "revision", &empty_layout, 128, 4);

    assert!(matches!(
        validation,
        Err(PersistentPromptCacheModelContractError::NoPersistentState)
    ));
}

fn block_token_count_sizing_inputs_fixture(
    maximum_context_token_count: usize,
    persistence_alignment_token_count: usize,
    sequence_state_payload_bytes_per_token: usize,
    boundary_state_payload_bytes: usize,
) -> BlockTokenCountSizingInputs {
    BlockTokenCountSizingInputs {
        maximum_context_token_count,
        persistence_alignment_token_count,
        sequence_state_payload_bytes_per_token,
        boundary_state_payload_bytes,
        global_ssd_quota_bytes: FITTING_GLOBAL_SSD_QUOTA_BYTES,
    }
}

#[test]
fn should_accept_an_aligned_configured_block_within_context() {
    let block_token_count = configured_or_derived_block_token_count(
        Some(16),
        &block_token_count_sizing_inputs_fixture(128, 4, 8, 32),
    )
    .expect("the aligned configured block should be accepted");

    assert_eq!(block_token_count, 16);
}

#[test]
fn should_reject_a_configured_block_that_breaks_persistence_alignment() {
    let block_token_count = configured_or_derived_block_token_count(
        Some(6),
        &block_token_count_sizing_inputs_fixture(128, 4, 8, 32),
    );

    assert!(matches!(
        block_token_count,
        Err(
            PersistentPromptCacheModelContractError::InvalidConfiguredBlockTokenCount {
                required_alignment_tokens: 4,
                maximum_context_tokens: 128,
            }
        )
    ));
}

#[test]
fn should_reject_a_zero_configured_block() {
    let block_token_count = configured_or_derived_block_token_count(
        Some(0),
        &block_token_count_sizing_inputs_fixture(128, 4, 8, 32),
    );

    assert!(matches!(
        block_token_count,
        Err(PersistentPromptCacheModelContractError::InvalidConfiguredBlockTokenCount { .. })
    ));
}

#[test]
fn should_reject_a_configured_block_above_the_context_limit() {
    let block_token_count = configured_or_derived_block_token_count(
        Some(256),
        &block_token_count_sizing_inputs_fixture(128, 4, 8, 32),
    );

    assert!(matches!(
        block_token_count,
        Err(
            PersistentPromptCacheModelContractError::InvalidConfiguredBlockTokenCount {
                maximum_context_tokens: 128,
                ..
            }
        )
    ));
}

#[test]
fn should_accept_a_configured_block_exactly_at_the_context_limit() {
    let block_token_count = configured_or_derived_block_token_count(
        Some(128),
        &block_token_count_sizing_inputs_fixture(128, 4, 8, 32),
    )
    .expect("the context-limit configured block should be accepted");

    assert_eq!(block_token_count, 128);
}

#[test]
fn should_derive_the_block_from_boundary_payload_geometry_when_not_configured() {
    let block_token_count = configured_or_derived_block_token_count(
        None,
        &block_token_count_sizing_inputs_fixture(128, 4, 8, 32),
    )
    .expect("the derived block should resolve");

    // The boundary payload amortization target is ceil(32 / 8) = 4 tokens.
    assert_eq!(block_token_count, 4);
}

fn block_file_sizing_inputs_fixture(
    decoder_cache_layout: &DecoderCacheLayout,
    global_ssd_quota_bytes: u64,
    block_token_count_is_user_configured: bool,
) -> BlockFileSizingInputs<'_> {
    BlockFileSizingInputs {
        decoder_cache_layout,
        maximum_context_token_count: FIXTURE_MAXIMUM_CONTEXT_TOKEN_COUNT,
        persistence_alignment_token_count: FIXTURE_PERSISTENCE_ALIGNMENT_TOKEN_COUNT,
        global_ssd_quota_bytes,
        block_token_count_is_user_configured,
        maximum_block_manifest_file_bytes: maximum_block_manifest_file_bytes(
            FIXTURE_MAXIMUM_CONTEXT_TOKEN_COUNT,
        )
        .expect("the fixture manifest size should resolve"),
    }
}

#[test]
fn should_keep_a_user_block_whose_chain_fits_the_ssd_quota() {
    let decoder_cache_layout = synthetic_hybrid_layout(8);

    let resolved_geometry = block_file_geometry_within_ssd_quota(
        16,
        &block_file_sizing_inputs_fixture(
            &decoder_cache_layout,
            FITTING_GLOBAL_SSD_QUOTA_BYTES,
            true,
        ),
    )
    .expect("a fitting user block should resolve");

    assert_eq!(resolved_geometry.block_token_count, 16);
    assert_eq!(
        resolved_geometry.sequence_state_file_bytes,
        exact_state_file_bytes(16, decoder_cache_layout.sequence_tensor_layouts())
            .expect("the fixture sequence file size should resolve")
    );
    assert_eq!(
        resolved_geometry.boundary_state_file_bytes,
        exact_state_file_bytes(16, decoder_cache_layout.boundary_tensor_layouts())
            .expect("the fixture boundary file size should resolve")
    );
    assert!(
        resolved_geometry.maximum_committed_block_bytes
            > resolved_geometry
                .sequence_state_file_bytes
                .saturating_add(resolved_geometry.boundary_state_file_bytes)
    );
}

#[test]
fn should_reject_a_user_block_whose_chain_exceeds_the_ssd_quota() {
    let decoder_cache_layout = synthetic_hybrid_layout(8);

    let resolved_geometry = block_file_geometry_within_ssd_quota(
        16,
        &block_file_sizing_inputs_fixture(&decoder_cache_layout, 1, true),
    );

    assert!(matches!(
        resolved_geometry,
        Err(
            PersistentPromptCacheModelContractError::ConfiguredBlockChainExceedsSsdQuota {
                configured_block_tokens: 16,
                ..
            }
        )
    ));
}

#[test]
fn should_grow_an_automatic_block_until_its_chain_fits_the_ssd_quota() {
    let decoder_cache_layout = synthetic_hybrid_layout(8);
    let fitting_geometry = block_file_geometry_within_ssd_quota(
        4,
        &block_file_sizing_inputs_fixture(
            &decoder_cache_layout,
            FITTING_GLOBAL_SSD_QUOTA_BYTES,
            false,
        ),
    )
    .expect("the unbounded quota should resolve the initial block");
    // A chain of 32 blocks of the initial four-token block just misses this
    // quota, while the grown eight-token block (16 blocks) still fits.
    let tight_global_ssd_quota_bytes = fitting_geometry.maximum_committed_block_bytes * 32 - 1;

    let grown_geometry = block_file_geometry_within_ssd_quota(
        4,
        &block_file_sizing_inputs_fixture(
            &decoder_cache_layout,
            tight_global_ssd_quota_bytes,
            false,
        ),
    )
    .expect("the grown block chain should fit the tight quota");

    assert_eq!(grown_geometry.block_token_count, 8);
    assert_eq!(
        grown_geometry.sequence_state_file_bytes,
        exact_state_file_bytes(8, decoder_cache_layout.sequence_tensor_layouts())
            .expect("the grown sequence file size should resolve")
    );
}

#[test]
fn should_fail_when_a_saturated_block_chain_still_exceeds_the_ssd_quota() {
    let decoder_cache_layout = synthetic_hybrid_layout(8);

    let resolved_geometry = block_file_geometry_within_ssd_quota(
        FIXTURE_MAXIMUM_CONTEXT_TOKEN_COUNT,
        &block_file_sizing_inputs_fixture(&decoder_cache_layout, 1, false),
    );

    // The block token count is pinned at the context limit, so automatic
    // growth saturates and the failure names the one-block files instead.
    assert!(matches!(
        resolved_geometry,
        Err(PersistentPromptCacheModelContractError::BlockFilesExceedSsdQuota { .. })
    ));
}

#[test]
fn should_take_the_largest_individual_tensor_as_publication_workspace() {
    // Each sequence tensor holds 4 bytes per token (2 elements, Float16), so
    // 16 tokens publish 64-byte tensors and beat the 32-byte boundary tensor.
    let workspace_bytes =
        direct_publication_workspace_bytes_for_block_token_count(&synthetic_hybrid_layout(8), 16)
            .expect("the hybrid workspace should derive");

    assert_eq!(workspace_bytes, 64);
}

#[test]
fn should_take_a_boundary_tensor_when_it_exceeds_every_sequence_tensor() {
    let workspace_bytes =
        direct_publication_workspace_bytes_for_block_token_count(&synthetic_hybrid_layout(64), 16)
            .expect("the boundary-dominated workspace should derive");

    assert_eq!(workspace_bytes, 256);
}

#[test]
fn should_fall_back_to_boundary_payloads_when_no_sequence_state_is_published() {
    let workspace_bytes = direct_publication_workspace_bytes_for_block_token_count(
        &synthetic_boundary_only_layout(8),
        16,
    )
    .expect("the sequence-free workspace should derive");

    assert_eq!(workspace_bytes, 32);
}

#[test]
fn should_derive_zero_sequence_workspace_for_a_zero_block_token_count() {
    let workspace_bytes =
        direct_publication_workspace_bytes_for_block_token_count(&synthetic_hybrid_layout(8), 0)
            .expect("the zero-length workspace should derive");

    assert_eq!(workspace_bytes, 32);
}

#[test]
fn should_accept_captured_state_within_both_budgets() {
    let validation = validated_capture_budgets(100, 200, 50, 100);

    assert!(matches!(validation, Ok(())));
}

#[test]
fn should_reject_capture_when_publication_workspace_exceeds_the_mlx_memory_ceiling() {
    let validation = validated_capture_budgets(201, 200, 50, 100);

    assert!(matches!(
        validation,
        Err(
            PersistentPromptCacheModelContractError::CaptureExceedsMlxMemoryCeiling {
                capture_memory_bytes: 201,
                effective_mlx_memory_ceiling_bytes: 200,
            }
        )
    ));
}

#[test]
fn should_reject_capture_when_one_committed_block_exceeds_the_ssd_quota() {
    let validation = validated_capture_budgets(100, 200, 101, 100);

    assert!(matches!(
        validation,
        Err(
            PersistentPromptCacheModelContractError::BlockFilesExceedSsdQuota {
                block_file_bytes: 101,
                global_ssd_quota_bytes: 100,
            }
        )
    ));
}

#[test]
fn should_accept_captured_state_exactly_at_both_budget_limits() {
    let validation = validated_capture_budgets(200, 200, 100, 100);

    assert!(matches!(validation, Ok(())));
}
