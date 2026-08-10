use std::fs::{self, File, OpenOptions};
use std::io::Write;

use astronomical_runtime_integration::{
    MlxDtype, MlxMemoryLimits, MlxRuntime, MlxRuntimeError, MlxSafetensorsWriterError,
};

#[test]
fn should_save_multiple_arrays_and_metadata_through_a_retained_file_descriptor() {
    let runtime = crate::common::runtime_test_support::runtime();
    let first_tensor = runtime
        .array_from_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 4])
        .expect("the test should create the first tensor");
    let second_tensor = runtime
        .zeros(&[2, 3], MlxDtype::BFloat16)
        .expect("the test should create the second tensor");
    let temporary_directory =
        tempfile::tempdir().expect("the test should create a temporary directory");
    let cache_file_path = temporary_directory.path().join("cache-block.safetensors");
    let cache_file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&cache_file_path)
        .expect("the test should create a new cache file");

    let write_outcome = runtime
        .save_safetensors(
            cache_file,
            &[
                ("layer_0_recurrent", &first_tensor),
                ("layer_1_keys", &second_tensor),
            ],
            &[("format_version", "1"), ("token_count", "2048")],
        )
        .expect("MLX should save tensors through the retained descriptor");
    assert_eq!(
        write_outcome.written_byte_count(),
        fs::metadata(&cache_file_path)
            .expect("the test should read the saved file metadata")
            .len()
    );

    let saved_cache_file =
        File::open(&cache_file_path).expect("the test should reopen the saved cache file");
    let saved_tensors = runtime
        .load_safetensors(saved_cache_file, None)
        .expect("MLX should load the file written through its official writer");
    let restored_first_tensor = saved_tensors
        .tensor("layer_0_recurrent")
        .expect("the first tensor should round trip");
    let restored_second_tensor = saved_tensors
        .tensor("layer_1_keys")
        .expect("the second tensor should round trip");

    assert_eq!(
        restored_first_tensor
            .to_vec_f32()
            .expect("the first tensor should evaluate"),
        vec![1.0, 2.0, 3.0, 4.0]
    );
    assert_eq!(restored_second_tensor.shape(), vec![2, 3]);
    assert_eq!(restored_second_tensor.dtype(), MlxDtype::BFloat16);
}

#[test]
fn should_preserve_descriptor_io_failure_separately_from_native_mlx_status() {
    let runtime = crate::common::runtime_test_support::runtime();
    let source_tensor = runtime
        .array_from_f32(&[1.0, 2.0], &[1, 2])
        .expect("the test should create a source tensor");
    let temporary_directory =
        tempfile::tempdir().expect("the test should create a temporary directory");
    let read_only_file_path = temporary_directory.path().join("read-only.safetensors");
    fs::write(&read_only_file_path, []).expect("the test should create the read-only target");
    let read_only_file = File::open(&read_only_file_path)
        .expect("the test should open the target without write access");

    let writer_error = runtime
        .save_safetensors(
            read_only_file,
            &[("sequence_state", &source_tensor)],
            &[("format_version", "test")],
        )
        .expect_err("descriptor write failure must be reported");

    assert!(matches!(
        writer_error,
        MlxSafetensorsWriterError::DescriptorIo { .. }
    ));
}

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

#[test]
fn should_serialize_safetensors_to_memory_before_background_file_writing() {
    let runtime = crate::common::runtime_test_support::runtime();
    let source_tensor = runtime
        .array_from_f32(&[2.0, 4.0, 6.0, 8.0], &[2, 2])
        .expect("the test should create the source tensor");

    let serialized_safetensors_bytes = runtime
        .serialize_safetensors(
            &[("layer_0_keys", &source_tensor)],
            &[("format_version", "test")],
        )
        .expect("MLX should serialize safetensors without filesystem access");

    let temporary_directory =
        tempfile::tempdir().expect("the test should create a temporary directory");
    let serialized_file_path = temporary_directory.path().join("serialized.safetensors");
    let mut serialized_file =
        File::create(&serialized_file_path).expect("the test should create the serialized file");
    serialized_file
        .write_all(&serialized_safetensors_bytes)
        .expect("the test should persist serialized bytes");
    drop(serialized_file);

    let restored_safetensors = runtime
        .load_safetensors(
            File::open(serialized_file_path).expect("the test should reopen serialized bytes"),
            None,
        )
        .expect("MLX should load the memory-serialized safetensors");
    assert_eq!(
        restored_safetensors
            .tensor("layer_0_keys")
            .expect("the serialized tensor should exist")
            .to_vec_f32()
            .expect("the serialized tensor should evaluate"),
        vec![2.0, 4.0, 6.0, 8.0]
    );
}

#[test]
fn should_bound_mlx_owned_safetensors_output_without_guessing_header_bytes() {
    let runtime = crate::common::runtime_test_support::runtime();
    let source_tensor = runtime
        .array_from_f32(&[1.0, 3.0, 5.0, 7.0], &[2, 2])
        .expect("the test should create a source tensor");
    let metadata_entries = [
        ("format_version", "11"),
        (
            "storage_contract_fingerprint",
            "0123456789abcdef0123456789abcdef",
        ),
    ];
    let unbounded_serialized_bytes = runtime
        .serialize_safetensors(
            &[("arbitrary.tensor_name", &source_tensor)],
            &metadata_entries,
        )
        .expect("MLX should serialize the arbitrary tensor");
    let exact_serialized_bytes = runtime
        .serialize_safetensors_with_maximum_byte_count(
            &[("arbitrary.tensor_name", &source_tensor)],
            &metadata_entries,
            unbounded_serialized_bytes.len(),
        )
        .expect("the exact MLX-produced byte count should fit");
    assert_eq!(exact_serialized_bytes, unbounded_serialized_bytes);

    let bounded_serialization_error = runtime
        .serialize_safetensors_with_maximum_byte_count(
            &[("arbitrary.tensor_name", &source_tensor)],
            &metadata_entries,
            unbounded_serialized_bytes.len() - 1,
        )
        .expect_err("one byte below the MLX-produced size must be rejected");
    assert!(matches!(
        bounded_serialization_error,
        MlxRuntimeError::SafetensorsSerializationLimitExceeded { .. }
    ));
}
