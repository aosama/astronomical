use astronomical_runtime_integration::{
    MlxDtype, MlxMemoryLimits, MlxMemorySnapshot, MlxRuntime, MlxRuntimeError,
};

const GRAPH_DIMENSION: i32 = 4096;
const ACTIVE_MEMORY_ENFORCEMENT_LIMIT_BYTES: usize = 32 * 1024 * 1024;
const ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES: usize = 0;

#[test]
#[ignore = "allocates and evaluates real MLX GPU arrays through the Rust MLX-C boundary"]
fn should_preserve_native_memory_transitions_through_the_mlx_c_boundary() {
    eprintln!("[mlx-c-memory-contract] status=start phase=runtime_initialization");
    let runtime = crate::common::runtime_test_support::runtime();
    eprintln!("[mlx-c-memory-contract] status=progress phase=clear_baseline_cache");
    runtime
        .clear_allocator_cache()
        .expect("the Rust MLX-C boundary should clear the allocator cache");
    runtime
        .reset_peak_memory()
        .expect("the Rust MLX-C boundary should reset peak memory");
    let before_graph = runtime
        .memory_snapshot()
        .expect("the Rust MLX-C boundary should read the baseline memory snapshot");
    print_memory_snapshot("baseline", &before_graph);

    let live_sum = {
        let left_array = runtime
            .zeros(&[GRAPH_DIMENSION, GRAPH_DIMENSION], MlxDtype::Float32)
            .expect("the Rust MLX-C boundary should construct the left lazy array");
        let right_array = runtime
            .zeros(&[GRAPH_DIMENSION, GRAPH_DIMENSION], MlxDtype::Float32)
            .expect("the Rust MLX-C boundary should construct the right lazy array");
        let lazy_sum = runtime
            .add(&left_array, &right_array)
            .expect("the Rust MLX-C boundary should construct the lazy sum");
        let lazy_sum_payload_byte_count = lazy_sum.byte_count();
        let after_lazy_graph = runtime
            .memory_snapshot()
            .expect("the Rust MLX-C boundary should read lazy graph memory");
        print_memory_snapshot("lazy_graph", &after_lazy_graph);
        assert!(
            after_lazy_graph.active_memory_bytes()
                < before_graph
                    .active_memory_bytes()
                    .saturating_add(lazy_sum_payload_byte_count),
            "lazy graph construction must not allocate the final sum payload"
        );
        assert_eq!(
            after_lazy_graph.allocator_cache_memory_bytes(),
            before_graph.allocator_cache_memory_bytes(),
            "lazy graph construction must not change reclaimable allocator-cache bytes"
        );
        eprintln!("[mlx-c-memory-contract] status=progress phase=evaluate_lazy_sum");
        runtime
            .evaluate_arrays(&[&lazy_sum])
            .expect("the Rust MLX-C boundary should evaluate the lazy sum");
        runtime
            .synchronize_gpu_stream()
            .expect("the Rust MLX-C boundary should synchronize the evaluated sum");
        let after_evaluation = runtime
            .memory_snapshot()
            .expect("the Rust MLX-C boundary should read evaluated memory");
        print_memory_snapshot("evaluated_sum", &after_evaluation);
        assert!(
            after_evaluation.active_memory_bytes() > before_graph.active_memory_bytes(),
            "evaluation should increase active MLX bytes"
        );
        assert!(
            after_evaluation.peak_memory_bytes() >= after_evaluation.active_memory_bytes(),
            "peak MLX bytes should include the live evaluated array"
        );
        runtime
            .reset_peak_memory()
            .expect("the Rust MLX-C boundary should reset peak memory with a live array");
        let after_peak_reset = runtime
            .memory_snapshot()
            .expect("the Rust MLX-C boundary should read memory after peak reset");
        print_memory_snapshot("peak_reset", &after_peak_reset);
        assert_eq!(after_peak_reset.peak_memory_bytes(), 0);
        assert!(
            after_peak_reset.active_memory_bytes() > before_graph.active_memory_bytes(),
            "peak reset should preserve non-baseline active residency for the live array"
        );
        lazy_sum
    };

    drop(live_sum);
    let after_owner_drop = runtime
        .memory_snapshot()
        .expect("the Rust MLX-C boundary should read memory after array owners drop");
    print_memory_snapshot("sum_owners_dropped", &after_owner_drop);
    assert_eq!(
        after_owner_drop.active_memory_bytes(),
        before_graph.active_memory_bytes(),
        "dropping all evaluated array owners should restore baseline active MLX bytes"
    );
    assert!(
        after_owner_drop.allocator_cache_memory_bytes() > 0,
        "dropping the evaluated array should retain reclaimable allocator bytes"
    );

    let async_array = runtime
        .zeros(&[GRAPH_DIMENSION, GRAPH_DIMENSION], MlxDtype::Float32)
        .expect("the Rust MLX-C boundary should construct an asynchronous array");
    eprintln!("[mlx-c-memory-contract] status=progress phase=submit_async_evaluation");
    runtime
        .async_eval_arrays(&[&async_array])
        .expect("the Rust MLX-C boundary should submit asynchronous evaluation");
    eprintln!("[mlx-c-memory-contract] status=progress phase=synchronize_async_cleanup");
    runtime
        .synchronize_gpu_stream_and_clear_allocator_cache()
        .expect("request-boundary cleanup should synchronize before clearing the cache");
    let after_synchronized_cleanup = runtime
        .memory_snapshot()
        .expect("the Rust MLX-C boundary should read memory after synchronized cleanup");
    print_memory_snapshot("async_live_after_cleanup", &after_synchronized_cleanup);
    assert_eq!(after_synchronized_cleanup.allocator_cache_memory_bytes(), 0);
    assert!(
        after_synchronized_cleanup.active_memory_bytes() > before_graph.active_memory_bytes(),
        "synchronized allocator cleanup must preserve the live asynchronous array"
    );
    drop(async_array);
    runtime
        .synchronize_gpu_stream_and_clear_allocator_cache()
        .expect("the final Rust MLX-C cleanup should complete");
    let after_final_owner_drop = runtime
        .memory_snapshot()
        .expect("the Rust MLX-C boundary should read memory after final owner drop");
    print_memory_snapshot("async_owners_dropped", &after_final_owner_drop);
    assert_eq!(
        after_final_owner_drop.active_memory_bytes(),
        before_graph.active_memory_bytes(),
        "dropping the final asynchronous array should restore baseline active MLX bytes"
    );
    assert_eq!(after_final_owner_drop.allocator_cache_memory_bytes(), 0);
    eprintln!("[mlx-c-memory-contract] status=success");
}

#[test]
#[ignore = "allocates and rejects real MLX GPU arrays through the Rust MLX-C boundary"]
fn should_map_new_and_cached_mlx_capacity_rejections_to_typed_errors() {
    eprintln!("[mlx-c-memory-contract] status=start phase=typed_capacity_rejection");
    let mut high_memory_runtime = MlxRuntime::initialize(
        MlxMemoryLimits::new(256 * 1024 * 1024, 256 * 1024 * 1024)
            .expect("the high test memory limits should be valid"),
    )
    .expect("the high-memory test runtime should initialize");
    high_memory_runtime
        .clear_allocator_cache()
        .expect("the high-memory test runtime should start with a clear allocator cache");
    let cached_array = high_memory_runtime
        .zeros(&[GRAPH_DIMENSION, GRAPH_DIMENSION], MlxDtype::Float32)
        .expect("the high-memory test runtime should create the cache fixture");
    high_memory_runtime
        .evaluate_arrays(&[&cached_array])
        .expect("the high-memory test runtime should evaluate the cache fixture");
    high_memory_runtime
        .synchronize_gpu_stream()
        .expect("the high-memory test runtime should synchronize the cache fixture");
    drop(cached_array);
    let cached_memory_snapshot = high_memory_runtime
        .memory_snapshot()
        .expect("the high-memory test runtime should report the cache fixture");
    assert!(cached_memory_snapshot.allocator_cache_memory_bytes() > 0);

    high_memory_runtime
        .update_memory_limits(
            MlxMemoryLimits::new(
                ACTIVE_MEMORY_ENFORCEMENT_LIMIT_BYTES,
                ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
            )
            .expect("the low test memory limits should be valid"),
        )
        .expect("the test runtime should accept the lower memory limits");

    let cached_rejection = evaluate_oversized_array(&high_memory_runtime)
        .expect_err("cached-buffer reuse should reject above the allowed active memory");
    assert_capacity_error(&cached_rejection);

    high_memory_runtime
        .clear_allocator_cache()
        .expect("capacity rejection cleanup should leave the runtime usable");
    let fitting_array = high_memory_runtime
        .zeros(&[1, 1], MlxDtype::Float32)
        .expect("a fitting allocation should remain usable after rejection");
    high_memory_runtime
        .evaluate_arrays(&[&fitting_array])
        .expect("a fitting evaluation should remain usable after rejection");
    eprintln!("[mlx-c-memory-contract] status=success phase=typed_capacity_rejection");
}

#[test]
#[ignore = "constructs a real MLX host-backed array through the Rust MLX-C boundary"]
fn should_preserve_host_backed_capacity_rejection_error_details() {
    eprintln!("[mlx-c-memory-contract] status=start phase=host_backed_capacity_rejection");
    let runtime = MlxRuntime::initialize(
        MlxMemoryLimits::new(
            ACTIVE_MEMORY_ENFORCEMENT_LIMIT_BYTES,
            ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
        )
        .expect("the low test memory limits should be valid"),
    )
    .expect("the low-memory test runtime should initialize");
    let host_values = vec![0.0_f32; (GRAPH_DIMENSION as usize) * (GRAPH_DIMENSION as usize)];
    let host_backed_rejection = runtime
        .array_from_f32(&host_values, &[GRAPH_DIMENSION, GRAPH_DIMENSION])
        .expect_err("host-backed allocation should reject above the allowed active memory");
    assert_capacity_error(&host_backed_rejection);
    eprintln!("[mlx-c-memory-contract] status=success phase=host_backed_capacity_rejection");
}

fn evaluate_oversized_array(runtime: &MlxRuntime) -> Result<(), MlxRuntimeError> {
    let oversized_array = runtime.zeros(&[GRAPH_DIMENSION, GRAPH_DIMENSION], MlxDtype::Float32)?;
    runtime.evaluate_arrays(&[&oversized_array])
}

fn assert_capacity_error(runtime_error: &MlxRuntimeError) {
    match runtime_error {
        MlxRuntimeError::ActiveMemoryLimitExceeded {
            active_memory_bytes,
            attempted_allocation_bytes,
            allowed_active_memory_bytes,
        } => {
            assert!(*attempted_allocation_bytes > *allowed_active_memory_bytes);
            assert!(*active_memory_bytes <= *allowed_active_memory_bytes);
            assert_eq!(
                *allowed_active_memory_bytes,
                ACTIVE_MEMORY_ENFORCEMENT_LIMIT_BYTES + ACTIVE_MEMORY_ENFORCEMENT_LIMIT_BYTES / 100
            );
        }
        other_error => panic!("expected typed MLX capacity error, got {other_error}"),
    }
}

fn print_memory_snapshot(phase_name: &str, memory_snapshot: &MlxMemorySnapshot) {
    eprintln!(
        "[mlx-c-memory-contract] status=progress phase={phase_name} active_bytes={} cache_bytes={} peak_bytes={}",
        memory_snapshot.active_memory_bytes(),
        memory_snapshot.allocator_cache_memory_bytes(),
        memory_snapshot.peak_memory_bytes(),
    );
}
