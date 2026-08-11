use std::fs::{File, OpenOptions};

use astronomical_runtime_integration::{MlxDtype, MlxMemoryLimits, MlxRuntime};

#[test]
fn should_write_unequal_noncontiguous_views_with_one_largest_tensor_workspace() {
    let runtime = crate::common::runtime_test_support::runtime();
    let original_memory_limits = runtime.memory_limits();
    let shared_backing = runtime
        .zeros(&[4_096, 4_096], MlxDtype::BFloat16)
        .expect("the test should create shared backing storage");
    runtime
        .evaluate_arrays(&[&shared_backing])
        .expect("the shared backing storage should materialize before setting the test ceiling");
    let largest_view = runtime
        .slice(&shared_backing, &[1, 1], &[2_049, 4_095], &[1, 1])
        .expect("the test should create the largest noncontiguous view");
    let medium_view = runtime
        .slice(&shared_backing, &[2_049, 1], &[3_585, 4_095], &[1, 1])
        .expect("the test should create the medium noncontiguous view");
    let smallest_view = runtime
        .slice(&shared_backing, &[3_073, 1], &[4_095, 4_095], &[1, 1])
        .expect("the test should create the smallest noncontiguous view");
    assert!(largest_view.byte_count() > medium_view.byte_count());
    assert!(medium_view.byte_count() > smallest_view.byte_count());
    let baseline_active_memory_bytes = runtime
        .memory_snapshot()
        .expect("the test should observe the materialized backing baseline")
        .active_memory_bytes();
    let active_memory_limit_bytes = baseline_active_memory_bytes
        .checked_add(largest_view.byte_count())
        .and_then(|bytes| bytes.checked_add(2_000_000))
        .expect("the test memory ceiling should fit usize");
    assert!(
        largest_view
            .byte_count()
            .saturating_add(medium_view.byte_count())
            > active_memory_limit_bytes.saturating_sub(baseline_active_memory_bytes),
        "the combined materializations must exceed the available workspace"
    );
    let constrained_memory_limits = MlxMemoryLimits::new(active_memory_limit_bytes, 0)
        .expect("the constrained writer memory limits should be valid");
    let mut guarded_runtime =
        RuntimeMemoryLimitGuard::new(runtime, original_memory_limits, constrained_memory_limits);
    let temporary_directory =
        tempfile::tempdir().expect("the test should create a temporary directory");
    let cache_file_path = temporary_directory
        .path()
        .join("streamed-views.safetensors");
    let cache_file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&cache_file_path)
        .expect("the test should create the streamed output file");

    guarded_runtime
        .runtime()
        .save_safetensors(
            cache_file,
            &[
                ("largest", &largest_view),
                ("medium", &medium_view),
                ("smallest", &smallest_view),
            ],
            &[("format_version", "streaming-workspace-test")],
        )
        .expect("one-at-a-time tensor materialization should fit the constrained ceiling");
    guarded_runtime.restore();
    drop(largest_view);
    drop(medium_view);
    drop(smallest_view);
    drop(shared_backing);

    let restored_safetensors = guarded_runtime
        .runtime()
        .load_safetensors(
            File::open(&cache_file_path).expect("the test should reopen the streamed output"),
            None,
        )
        .expect("the streamed views should reload");
    assert_eq!(
        restored_safetensors
            .tensor("largest")
            .expect("the largest view should exist")
            .shape(),
        vec![2_048, 4_094]
    );
    assert_eq!(
        restored_safetensors
            .tensor("medium")
            .expect("the medium view should exist")
            .shape(),
        vec![1_536, 4_094]
    );
    assert_eq!(
        restored_safetensors
            .tensor("smallest")
            .expect("the smallest view should exist")
            .shape(),
        vec![1_022, 4_094]
    );
}

struct RuntimeMemoryLimitGuard {
    runtime: MlxRuntime,
    original_memory_limits: MlxMemoryLimits,
    has_restored_original_limits: bool,
}

impl RuntimeMemoryLimitGuard {
    fn new(
        mut runtime: MlxRuntime,
        original_memory_limits: MlxMemoryLimits,
        constrained_memory_limits: MlxMemoryLimits,
    ) -> Self {
        runtime
            .update_memory_limits(constrained_memory_limits)
            .expect("the test should install the constrained memory ceiling");
        Self {
            runtime,
            original_memory_limits,
            has_restored_original_limits: false,
        }
    }

    fn runtime(&self) -> &MlxRuntime {
        &self.runtime
    }

    fn restore(&mut self) {
        if self.has_restored_original_limits {
            return;
        }
        self.runtime
            .update_memory_limits(self.original_memory_limits)
            .expect("the test should restore the original memory limits");
        self.has_restored_original_limits = true;
    }
}

impl Drop for RuntimeMemoryLimitGuard {
    fn drop(&mut self) {
        if !self.has_restored_original_limits {
            let _restore_result = self
                .runtime
                .update_memory_limits(self.original_memory_limits);
        }
        let _cleanup_result = self
            .runtime
            .synchronize_gpu_stream_and_clear_allocator_cache();
    }
}
