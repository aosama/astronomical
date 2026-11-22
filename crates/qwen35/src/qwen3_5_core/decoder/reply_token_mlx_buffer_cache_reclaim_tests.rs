use mlx_bridge::MlxDtype;
use runtime_integration::{MLX_BUFFER_CACHE_RECLAIM_THRESHOLD_BYTES, MlxMemoryLimits, MlxRuntime};
use serial_test::serial;

use super::reply_token_mlx_buffer_cache_reclaim::{
    DRAFT_ROUND_MLX_BUFFER_CACHE_RECLAIM_THRESHOLD_BYTES, OutgrownAttentionSlabReclaim,
    reclaim_mlx_buffer_cache_between_draft_rounds,
};

const TEST_MLX_MEMORY_LIMIT_BYTES: usize = 4 * 1024 * 1024 * 1024;
const FLOAT32_BYTES: usize = 4;
const DECODER_STATE_BYTES_BEFORE_GROWTH: u64 = 800_000_000;
const DECODER_STATE_BYTES_AFTER_GROWTH: u64 = 805_000_000;

/// A tracker that has already seen the state at its size before growth, as
/// a request has after its first reply token.
fn reclaim_that_saw_the_state_before_growth() -> OutgrownAttentionSlabReclaim {
    let mut outgrown_attention_slab_reclaim = OutgrownAttentionSlabReclaim::default();
    outgrown_attention_slab_reclaim.observe_slab_growth(DECODER_STATE_BYTES_BEFORE_GROWTH);
    outgrown_attention_slab_reclaim
}

fn test_runtime() -> MlxRuntime {
    MlxRuntime::initialize(
        MlxMemoryLimits::new(TEST_MLX_MEMORY_LIMIT_BYTES, TEST_MLX_MEMORY_LIMIT_BYTES)
            .expect("the reclaim test memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize")
}

/// Allocates one slab of `slab_bytes` and frees it, which is what an
/// outgrown attention slab does: the bytes land in the `mlx_buffer_cache`.
fn strand_one_slab_in_the_mlx_buffer_cache(runtime: &MlxRuntime, slab_bytes: usize) {
    let slab_element_count =
        i32::try_from(slab_bytes / FLOAT32_BYTES).expect("the test slab fits an MLX dimension");
    let outgrown_slab = runtime
        .zeros(&[slab_element_count], MlxDtype::Float32)
        .expect("the slab builds");
    runtime
        .evaluate_arrays(&[&outgrown_slab])
        .expect("the slab evaluates");
    drop(outgrown_slab);
    // Metal hands a buffer back only once the command that used it has
    // completed, so the freed slab is counted after the stream drains.
    runtime.synchronize_gpu_stream().expect("the stream drains");
}

fn mlx_buffer_cache_bytes(runtime: &MlxRuntime) -> usize {
    runtime
        .memory_snapshot()
        .expect("the allocator reports its memory")
        .mlx_buffer_cache_bytes()
}

#[test]
#[serial]
fn should_release_the_mlx_buffer_cache_once_an_outgrown_slab_passes_the_threshold() {
    let runtime = test_runtime();
    runtime
        .synchronize_gpu_stream_and_clear_mlx_buffer_cache()
        .expect("the test starts from an empty mlx_buffer_cache");
    strand_one_slab_in_the_mlx_buffer_cache(&runtime, MLX_BUFFER_CACHE_RECLAIM_THRESHOLD_BYTES);
    assert!(mlx_buffer_cache_bytes(&runtime) >= MLX_BUFFER_CACHE_RECLAIM_THRESHOLD_BYTES);

    reclaim_that_saw_the_state_before_growth()
        .reclaim_after_slab_growth(&runtime, DECODER_STATE_BYTES_AFTER_GROWTH)
        .expect("the release succeeds");

    assert_eq!(mlx_buffer_cache_bytes(&runtime), 0);
}

#[test]
#[serial]
fn should_keep_the_mlx_buffer_cache_while_it_is_below_the_threshold() {
    let runtime = test_runtime();
    runtime
        .synchronize_gpu_stream_and_clear_mlx_buffer_cache()
        .expect("the test starts from an empty mlx_buffer_cache");
    let reusable_working_buffer_bytes = MLX_BUFFER_CACHE_RECLAIM_THRESHOLD_BYTES / 2;
    strand_one_slab_in_the_mlx_buffer_cache(&runtime, reusable_working_buffer_bytes);
    let mlx_buffer_cache_bytes_before = mlx_buffer_cache_bytes(&runtime);
    assert!(mlx_buffer_cache_bytes_before >= reusable_working_buffer_bytes);

    reclaim_that_saw_the_state_before_growth()
        .reclaim_after_slab_growth(&runtime, DECODER_STATE_BYTES_AFTER_GROWTH)
        .expect("the check succeeds");

    assert_eq!(
        mlx_buffer_cache_bytes(&runtime),
        mlx_buffer_cache_bytes_before
    );
}

#[test]
#[serial]
fn should_succeed_on_an_empty_mlx_buffer_cache() {
    let runtime = test_runtime();
    runtime
        .synchronize_gpu_stream_and_clear_mlx_buffer_cache()
        .expect("the test starts from an empty mlx_buffer_cache");

    reclaim_that_saw_the_state_before_growth()
        .reclaim_after_slab_growth(&runtime, DECODER_STATE_BYTES_AFTER_GROWTH)
        .expect("nothing to release is fine");

    assert_eq!(mlx_buffer_cache_bytes(&runtime), 0);
}

#[test]
#[serial]
fn should_keep_working_buffers_above_the_threshold_when_no_slab_grew() {
    let runtime = test_runtime();
    runtime
        .synchronize_gpu_stream_and_clear_mlx_buffer_cache()
        .expect("the test starts from an empty mlx_buffer_cache");
    strand_one_slab_in_the_mlx_buffer_cache(&runtime, MLX_BUFFER_CACHE_RECLAIM_THRESHOLD_BYTES);
    let mlx_buffer_cache_bytes_before = mlx_buffer_cache_bytes(&runtime);

    reclaim_that_saw_the_state_before_growth()
        .reclaim_after_slab_growth(&runtime, DECODER_STATE_BYTES_BEFORE_GROWTH)
        .expect("the check succeeds");

    assert_eq!(
        mlx_buffer_cache_bytes(&runtime),
        mlx_buffer_cache_bytes_before
    );
}

#[test]
fn should_report_growth_only_when_the_state_became_larger() {
    let mut outgrown_attention_slab_reclaim = OutgrownAttentionSlabReclaim::default();
    assert!(!outgrown_attention_slab_reclaim.observe_slab_growth(0));
    assert!(outgrown_attention_slab_reclaim.observe_slab_growth(DECODER_STATE_BYTES_BEFORE_GROWTH));
    assert!(
        !outgrown_attention_slab_reclaim.observe_slab_growth(DECODER_STATE_BYTES_BEFORE_GROWTH)
    );
    assert!(outgrown_attention_slab_reclaim.observe_slab_growth(DECODER_STATE_BYTES_AFTER_GROWTH));
    // A drafted rewind shrinks the state and must not count as growth.
    assert!(
        !outgrown_attention_slab_reclaim.observe_slab_growth(DECODER_STATE_BYTES_BEFORE_GROWTH)
    );
    assert!(outgrown_attention_slab_reclaim.observe_slab_growth(DECODER_STATE_BYTES_AFTER_GROWTH));
}

#[test]
#[serial]
fn should_release_between_draft_rounds_above_the_threshold_without_any_slab_growth() {
    let runtime = test_runtime();
    runtime
        .synchronize_gpu_stream_and_clear_mlx_buffer_cache()
        .expect("the test starts from an empty mlx_buffer_cache");
    strand_one_slab_in_the_mlx_buffer_cache(
        &runtime,
        DRAFT_ROUND_MLX_BUFFER_CACHE_RECLAIM_THRESHOLD_BYTES,
    );

    reclaim_mlx_buffer_cache_between_draft_rounds(&runtime).expect("the release succeeds");

    assert_eq!(mlx_buffer_cache_bytes(&runtime), 0);
}

#[test]
#[serial]
fn should_keep_a_small_mlx_buffer_cache_between_draft_rounds() {
    let runtime = test_runtime();
    runtime
        .synchronize_gpu_stream_and_clear_mlx_buffer_cache()
        .expect("the test starts from an empty mlx_buffer_cache");
    strand_one_slab_in_the_mlx_buffer_cache(
        &runtime,
        DRAFT_ROUND_MLX_BUFFER_CACHE_RECLAIM_THRESHOLD_BYTES / 2,
    );
    let mlx_buffer_cache_bytes_before = mlx_buffer_cache_bytes(&runtime);
    assert!(mlx_buffer_cache_bytes_before > 0);

    reclaim_mlx_buffer_cache_between_draft_rounds(&runtime).expect("the check succeeds");

    assert_eq!(
        mlx_buffer_cache_bytes(&runtime),
        mlx_buffer_cache_bytes_before
    );
}
