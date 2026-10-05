//! Laguna-owned capture/restore bridge journeys for persistent prompt-cache
//! blocks, driven entirely by synthetic fixtures.
//!
//! The neutral disk store, block manifests, and storage-contract fingerprints
//! already prove model/revision/ancestry mismatch rejection in the shared
//! persistent-cache tests. This file proves what only Laguna owns: the mixed
//! append-only plus rotating state must round-trip exactly through
//! `extract_cache_block_tensors` and the incremental block restore, and every
//! corrupted block variant must fail loudly instead of restoring stale state.

use std::collections::HashMap;

use astronomical_model_serving::{LagunaDecoderState, LagunaModel, PerformanceAttribution};
use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};

use crate::common::laguna;
use crate::common::{
    DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES, DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
};
use astronomical_mlx_c_rust::{MlxArray, MlxDtype};

const BLOCK_TOKEN_COUNT: usize = 2;
/// Tiny fixtures round through MLX allocation granularity, so memory-bound
/// asserts carry explicit headroom instead of exact equality.
const ALLOCATION_GRANULARITY_HEADROOM_BYTES: u64 = 2 * 1024 * 1024;

type TensorMap = HashMap<String, MlxArray>;

fn test_runtime() -> MlxRuntime {
    MlxRuntime::initialize(
        MlxMemoryLimits::new(
            DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES,
            DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
        )
        .expect("Laguna bridge test memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize")
}

fn mixed_model(runtime: &MlxRuntime) -> LagunaModel {
    let contract = laguna::tiny_mixed_contract();
    let weights = laguna::bind_tiny_weights(runtime, &contract);
    LagunaModel::new(
        contract,
        weights,
        crate::common::test_worker_kernel_capabilities(runtime),
    )
    .expect("the tiny mixed Laguna model should construct")
}

/// Runs one four-token prefill so both state kinds sit exactly on the second
/// block boundary, then captures each two-token sequence block plus the newest
/// boundary snapshot from the populated state.
fn capture_two_blocks(
    runtime: &MlxRuntime,
    model: &LagunaModel,
    decoder_state: &mut LagunaDecoderState,
    performance_attribution: &mut PerformanceAttribution,
) -> (TensorMap, TensorMap, TensorMap) {
    let prompt_tokens = runtime
        .array_from_u32(&[1, 2, 3, 4], &[4])
        .expect("the four-token prompt should build");
    model
        .forward(
            runtime,
            &prompt_tokens,
            decoder_state,
            performance_attribution,
        )
        .expect("the four-token prefill should execute");
    let (first_sequence_block, _) = decoder_state
        .extract_cache_block_tensors(runtime, 0, BLOCK_TOKEN_COUNT)
        .expect("the first complete block should capture");
    let (second_sequence_block, boundary_snapshot) = decoder_state
        .extract_cache_block_tensors(runtime, BLOCK_TOKEN_COUNT, 2 * BLOCK_TOKEN_COUNT)
        .expect("the second complete block should capture");
    (
        first_sequence_block,
        second_sequence_block,
        boundary_snapshot,
    )
}

fn fresh_captured_sequence_block(
    runtime: &MlxRuntime,
    model: &LagunaModel,
    performance_attribution: &mut PerformanceAttribution,
) -> TensorMap {
    let mut source_state = LagunaDecoderState::empty(model.contract())
        .expect("the capture source state should allocate");
    let (first_sequence_block, _, _) =
        capture_two_blocks(runtime, model, &mut source_state, performance_attribution);
    first_sequence_block
}

fn fresh_boundary_snapshot(
    runtime: &MlxRuntime,
    model: &LagunaModel,
    performance_attribution: &mut PerformanceAttribution,
) -> TensorMap {
    let mut source_state = LagunaDecoderState::empty(model.contract())
        .expect("the snapshot source state should allocate");
    let (_, _, boundary_snapshot) =
        capture_two_blocks(runtime, model, &mut source_state, performance_attribution);
    boundary_snapshot
}

fn fresh_state(model: &LagunaModel) -> LagunaDecoderState {
    LagunaDecoderState::empty(model.contract()).expect("the fresh decoder state should allocate")
}

fn tensor_map_byte_count(tensors: &TensorMap) -> u64 {
    tensors
        .values()
        .map(|tensor| u64::try_from(tensor.byte_count()).unwrap_or(u64::MAX))
        .fold(0_u64, u64::saturating_add)
}

#[tokio::test]
async fn should_round_trip_a_synthetic_mixed_layer_ordering_through_capture_and_restore() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = test_runtime();
    let model = mixed_model(&runtime);
    let mut performance_attribution = PerformanceAttribution::enabled();

    let mut original_state = fresh_state(&model);
    let (first_sequence_block, second_sequence_block, mut boundary_snapshot) = capture_two_blocks(
        &runtime,
        &model,
        &mut original_state,
        &mut performance_attribution,
    );

    let mut restored_state = fresh_state(&model);
    let sequence_blocks = vec![first_sequence_block, second_sequence_block];
    restored_state
        .restore_sequence_blocks_from_concat(&runtime, &sequence_blocks, 2 * BLOCK_TOKEN_COUNT)
        .expect("the captured blocks should restore into a fresh state");
    restored_state
        .restore_boundary_snapshot(&runtime, &mut boundary_snapshot)
        .expect("the newest rotating snapshot should restore");

    // The synthetic mixed ordering must round-trip with exact per-layer state:
    // the append-only layer at the block boundary, and the rotating layer with
    // its committed window, absolute position, and ring write index intact.
    assert_eq!(
        restored_state.absolute_position(0),
        original_state.absolute_position(0),
        "the append-only layer must resume at the captured absolute position"
    );
    assert_eq!(
        restored_state.committed_token_count(1),
        original_state.committed_token_count(1),
        "the rotating layer must restore its committed token count"
    );
    assert_eq!(
        restored_state.ring_write_index(1),
        original_state.ring_write_index(1),
        "the rotating layer must restore its ring write index"
    );

    // Lossless continuation: after restore, the same next token must produce
    // bit-identical logits from both states.
    let next_tokens = runtime
        .array_from_u32(&[5], &[1])
        .expect("the continuation token should build");
    let original_logits = model
        .forward(
            &runtime,
            &next_tokens,
            &mut original_state,
            &mut performance_attribution,
        )
        .expect("the original continuation should execute");
    let restored_logits = model
        .forward(
            &runtime,
            &next_tokens,
            &mut restored_state,
            &mut performance_attribution,
        )
        .expect("the restored continuation should execute");
    assert_eq!(
        original_logits.to_vec_f32().expect("original logits"),
        restored_logits.to_vec_f32().expect("restored logits"),
        "a restored state must continue generation bit-identically"
    );
}

#[tokio::test]
async fn should_restore_from_concat_within_destination_plus_source_bound() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = test_runtime();
    let model = mixed_model(&runtime);
    let mut performance_attribution = PerformanceAttribution::enabled();

    let mut original_state = fresh_state(&model);
    let (first_sequence_block, second_sequence_block, mut boundary_snapshot) = capture_two_blocks(
        &runtime,
        &model,
        &mut original_state,
        &mut performance_attribution,
    );
    let sequence_payload_bytes = tensor_map_byte_count(&first_sequence_block)
        .saturating_add(tensor_map_byte_count(&second_sequence_block));
    let snapshot_payload_bytes = tensor_map_byte_count(&boundary_snapshot);

    let mut restored_state = fresh_state(&model);
    let baseline_active_bytes = runtime
        .memory_snapshot()
        .expect("active memory should be sampled before the restore")
        .active_memory_bytes() as u64;
    runtime
        .reset_peak_memory()
        .expect("the MLX peak counter should reset so the restore peak is measured alone");
    let sequence_blocks = vec![first_sequence_block, second_sequence_block];
    restored_state
        .restore_sequence_blocks_from_concat(&runtime, &sequence_blocks, 2 * BLOCK_TOKEN_COUNT)
        .expect("the captured blocks should restore in one concat per tensor");
    restored_state
        .restore_boundary_snapshot(&runtime, &mut boundary_snapshot)
        .expect("the newest rotating snapshot should complete the restore");

    // The concat restore must release its transient back to the baseline:
    // baseline plus the seated destination plus the restored snapshot. Tiny
    // fixtures round through MLX allocation granularity, so the bound carries
    // explicit headroom instead of exact byte equality.
    let after_restore_active_bytes = runtime
        .memory_snapshot()
        .expect("active memory should be sampled after the restore")
        .active_memory_bytes() as u64;
    let return_to_baseline_bound_bytes = baseline_active_bytes
        .saturating_add(sequence_payload_bytes.saturating_mul(2))
        .saturating_add(snapshot_payload_bytes)
        .saturating_add(ALLOCATION_GRANULARITY_HEADROOM_BYTES);
    assert!(
        after_restore_active_bytes <= return_to_baseline_bound_bytes,
        "the concat restore must release its transient back to the baseline: \
         after={after_restore_active_bytes} bound={return_to_baseline_bound_bytes} \
         baseline={baseline_active_bytes} sequence={sequence_payload_bytes} \
         snapshot={snapshot_payload_bytes}"
    );
}

#[tokio::test]
async fn should_reject_bridge_mismatches_instead_of_restoring_stale_state() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = test_runtime();
    let model = mixed_model(&runtime);
    let mut performance_attribution = PerformanceAttribution::enabled();

    // An empty state cannot capture: the append-only layer holds no keys yet.
    let empty_state = fresh_state(&model);
    let capture_rejection = empty_state
        .extract_cache_block_tensors(&runtime, 0, BLOCK_TOKEN_COUNT)
        .expect_err("capturing from an empty append-only layer must fail");
    assert!(
        format!("{capture_rejection:?}").contains("missing keys during capture"),
        "the capture rejection should name the missing append-only keys: {capture_rejection:?}"
    );

    // A sequence block missing one tensor must fail before the layer restores.
    let mut incomplete_first_block =
        fresh_captured_sequence_block(&runtime, &model, &mut performance_attribution);
    incomplete_first_block
        .remove("layer_0_attention.keys")
        .expect("the captured keys tensor should be present");
    let sequence_blocks = vec![incomplete_first_block];
    let missing_tensor_rejection = fresh_state(&model)
        .restore_sequence_blocks_from_concat(&runtime, &sequence_blocks, BLOCK_TOKEN_COUNT)
        .expect_err("a sequence block missing a tensor must fail");
    assert!(
        format!("{missing_tensor_rejection:?}").contains("missing a tensor"),
        "the rejection should name the missing sequence tensor: {missing_tensor_rejection:?}"
    );

    // A boundary snapshot missing the rotating keys must fail.
    let mut keys_missing_snapshot =
        fresh_boundary_snapshot(&runtime, &model, &mut performance_attribution);
    keys_missing_snapshot
        .remove("layer_1_attention.keys")
        .expect("the rotating keys should be present");
    let mut snapshot_only_state = fresh_state(&model);
    let snapshot_only_block = vec![fresh_captured_sequence_block(
        &runtime,
        &model,
        &mut performance_attribution,
    )];
    snapshot_only_state
        .restore_sequence_blocks_from_concat(&runtime, &snapshot_only_block, BLOCK_TOKEN_COUNT)
        .expect("the captured block should seat the append-only layers");
    let missing_keys_rejection = snapshot_only_state
        .restore_boundary_snapshot(&runtime, &mut keys_missing_snapshot)
        .expect_err("a snapshot missing rotating keys must fail");
    assert!(
        format!("{missing_keys_rejection:?}").contains("missing keys"),
        "the rejection should name the missing rotating keys: {missing_keys_rejection:?}"
    );

    // A boundary snapshot missing a rotating counter must fail.
    let mut counter_missing_snapshot =
        fresh_boundary_snapshot(&runtime, &model, &mut performance_attribution);
    counter_missing_snapshot
        .remove("layer_1_attention.absolute_position")
        .expect("the absolute position counter should be present");
    let mut counter_only_state = fresh_state(&model);
    let counter_only_block = vec![fresh_captured_sequence_block(
        &runtime,
        &model,
        &mut performance_attribution,
    )];
    counter_only_state
        .restore_sequence_blocks_from_concat(&runtime, &counter_only_block, BLOCK_TOKEN_COUNT)
        .expect("the captured block should seat the append-only layers");
    let missing_counter_rejection = counter_only_state
        .restore_boundary_snapshot(&runtime, &mut counter_missing_snapshot)
        .expect_err("a snapshot missing a rotating counter must fail");
    assert!(
        format!("{missing_counter_rejection:?}").contains("counter tensor is missing"),
        "the rejection should name the missing counter: {missing_counter_rejection:?}"
    );

    // A counter stored with a foreign dtype must fail before restoration.
    let mut foreign_dtype_snapshot =
        fresh_boundary_snapshot(&runtime, &model, &mut performance_attribution);
    let float16_counter_array = foreign_dtype_snapshot
        .remove("layer_1_attention.ring_write_index")
        .expect("the ring write index counter should be present");
    let float16_counter = runtime
        .astype(&float16_counter_array, MlxDtype::Float16)
        .expect("the float16 counter should build");
    foreign_dtype_snapshot.insert(
        "layer_1_attention.ring_write_index".to_owned(),
        float16_counter,
    );
    let mut dtype_only_state = fresh_state(&model);
    let dtype_only_block = vec![fresh_captured_sequence_block(
        &runtime,
        &model,
        &mut performance_attribution,
    )];
    dtype_only_state
        .restore_sequence_blocks_from_concat(&runtime, &dtype_only_block, BLOCK_TOKEN_COUNT)
        .expect("the captured block should seat the append-only layers");
    let foreign_dtype_rejection = dtype_only_state
        .restore_boundary_snapshot(&runtime, &mut foreign_dtype_snapshot)
        .expect_err("a non-float32 rotating counter must fail");
    assert!(
        format!("{foreign_dtype_rejection:?}").contains("float32"),
        "the rejection should name the counter dtype: {foreign_dtype_rejection:?}"
    );

    // A zero absolute position claims a restore with no live tokens.
    let mut zero_position_snapshot =
        fresh_boundary_snapshot(&runtime, &model, &mut performance_attribution);
    let zero_counter = runtime
        .array_from_f32(&[0.0], &[1])
        .expect("the zero counter should build");
    zero_position_snapshot.insert(
        "layer_1_attention.absolute_position".to_owned(),
        zero_counter,
    );
    let mut zero_position_state = fresh_state(&model);
    let zero_position_block = vec![fresh_captured_sequence_block(
        &runtime,
        &model,
        &mut performance_attribution,
    )];
    zero_position_state
        .restore_sequence_blocks_from_concat(&runtime, &zero_position_block, BLOCK_TOKEN_COUNT)
        .expect("the captured block should seat the append-only layers");
    let zero_position_rejection = zero_position_state
        .restore_boundary_snapshot(&runtime, &mut zero_position_snapshot)
        .expect_err("a restore claiming zero live tokens must fail");
    assert!(
        format!("{zero_position_rejection:?}").contains("at least one token"),
        "the rejection should name the empty rotating restore: {zero_position_rejection:?}"
    );
}
