use std::time::Duration;

use astronomical_model_serving::{
    PerformanceAttribution, PerformanceOperation, PersistentPromptCacheBlockKey,
    PersistentPromptCachePublicationOutcome,
};
use astronomical_runtime_integration::{MlxDtype, MlxMemoryLimits, MlxRuntime};

use super::persistent_prompt_cache_disk_store_support::{
    block_tokens_for_seed, open_persistent_prompt_cache_disk_store,
    persistent_prompt_cache_block_key_for_seed, runtime_with_shared_limits,
    synthetic_kv_block_tensors, synthetic_recurrent_snapshot_tensors,
};
use crate::common::qwen3_5_moe::persistent_prompt_cache_model_contract;

const LARGE_CACHE_LIMIT_BYTES: u64 = 10 * 1024 * 1024 * 1024;
const DIRECT_MLX_PROMPT_CACHE_TEST_TIMEOUT: Duration = Duration::from_secs(115);

#[tokio::test]
async fn should_publish_required_capture_directly_without_serialized_host_ownership() {
    tokio::time::timeout(DIRECT_MLX_PROMPT_CACHE_TEST_TIMEOUT, async {
        let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
        let runtime = runtime_with_shared_limits();
        let persistent_prompt_cache_directory =
            tempfile::tempdir().expect("the test should create a prompt-cache directory");
        let persistent_prompt_cache = open_persistent_prompt_cache_disk_store(
            &persistent_prompt_cache_directory,
            LARGE_CACHE_LIMIT_BYTES,
        )
        .expect("the persistent prompt cache should open");
        let sequence_state_tensors = synthetic_kv_block_tensors(&runtime);
        let boundary_state_tensors = synthetic_recurrent_snapshot_tensors(&runtime);
        let mut performance_attribution = PerformanceAttribution::enabled();

        let publication_outcome = persistent_prompt_cache
            .publish_block_with_performance_attribution(
                &runtime,
                &persistent_prompt_cache_block_key_for_seed(0),
                None,
                &sequence_state_tensors,
                &boundary_state_tensors,
                &mut performance_attribution,
            )
            .expect("the required capture should publish directly");

        assert_eq!(
            publication_outcome,
            PersistentPromptCachePublicationOutcome::Published
        );
        assert!(
            performance_attribution
                .operation_measurement(
                    PerformanceOperation::PersistentPromptCacheKvBlockSerialization,
                )
                .is_some(),
            "direct publication should retain write attribution"
        );
        assert!(
            performance_attribution
                .operation_measurement(
                    PerformanceOperation::PersistentPromptCachePublicationSynchronizationWait,
                )
                .is_some(),
            "direct publication should attribute durable synchronization"
        );
        assert!(
            performance_attribution
                .operation_measurement(
                    PerformanceOperation::PersistentPromptCachePublicationValidation,
                )
                .is_some(),
            "direct publication should attribute contract validation"
        );
        assert!(
            performance_attribution
                .operation_measurement(
                    PerformanceOperation::PersistentPromptCacheGlobalQuotaEviction,
                )
                .is_some(),
            "direct publication should attribute global quota eviction work"
        );
        assert!(
            performance_attribution
                .operation_measurement(PerformanceOperation::PersistentPromptCacheAtomicCommit)
                .is_some(),
            "direct publication should attribute its atomic commit"
        );
    })
    .await
    .expect("the direct publication test should finish within 115 seconds");
}

#[tokio::test]
async fn should_publish_four_production_sized_boundaries_sequentially() {
    tokio::time::timeout(DIRECT_MLX_PROMPT_CACHE_TEST_TIMEOUT, async {
        let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
        let runtime = runtime_with_shared_limits();
        let persistent_prompt_cache_directory =
            tempfile::tempdir().expect("the test should create a prompt-cache directory");
        let persistent_prompt_cache = open_persistent_prompt_cache_disk_store(
            &persistent_prompt_cache_directory,
            LARGE_CACHE_LIMIT_BYTES,
        )
        .expect("the persistent prompt cache should open");
        let sequence_state_tensors = synthetic_kv_block_tensors(&runtime);
        let boundary_state_tensors = synthetic_recurrent_snapshot_tensors(&runtime);
        let mut parent_block_key: Option<PersistentPromptCacheBlockKey> = None;

        for boundary_index in 0_u32..4 {
            let block_key = match parent_block_key.as_ref() {
                None => persistent_prompt_cache_block_key_for_seed(0),
                Some(parent_block_key) => parent_block_key
                    .for_child_block(&block_tokens_for_seed(boundary_index * 10_000))
                    .expect("the test should construct the next child block identity"),
            };
            let publication_outcome = persistent_prompt_cache
                .publish_block(
                    &runtime,
                    &block_key,
                    parent_block_key.as_ref(),
                    &sequence_state_tensors,
                    &boundary_state_tensors,
                )
                .expect("every completed boundary should publish");
            assert_eq!(
                publication_outcome,
                PersistentPromptCachePublicationOutcome::Published
            );
            parent_block_key = Some(block_key);
        }

        assert_eq!(persistent_prompt_cache.sequence_state_block_count(), 4);
    })
    .await
    .expect("the sequential direct publication test should finish within 115 seconds");
}

#[tokio::test]
async fn should_retry_the_same_captured_arrays_after_releasing_artificial_active_pressure() {
    tokio::time::timeout(DIRECT_MLX_PROMPT_CACHE_TEST_TIMEOUT, async {
        let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
        let runtime = runtime_with_shared_limits();
        let original_memory_limits = runtime.memory_limits();
        let persistent_prompt_cache_directory =
            tempfile::tempdir().expect("the test should create a prompt-cache directory");
        let persistent_prompt_cache = open_persistent_prompt_cache_disk_store(
            &persistent_prompt_cache_directory,
            LARGE_CACHE_LIMIT_BYTES,
        )
        .expect("the persistent prompt cache should open");
        let block_key = persistent_prompt_cache_block_key_for_seed(0);
        let mut sequence_state_tensors = synthetic_kv_block_tensors(&runtime);
        let boundary_state_tensors = synthetic_recurrent_snapshot_tensors(&runtime);
        let model_contract = persistent_prompt_cache_model_contract();
        let sequence_tensor_layout = model_contract
            .decoder_cache_layout()
            .sequence_tensor_layouts()
            .into_iter()
            .next()
            .expect("the production contract should contain sequence state");
        let tensor_layout = sequence_tensor_layout.tensor_layout();
        let expected_dimensions = tensor_layout
            .dimensions()
            .iter()
            .enumerate()
            .map(|(dimension_index, dimension)| {
                if tensor_layout.sequence_axis() == Some(dimension_index) {
                    model_contract.block_token_count() as i32
                } else {
                    *dimension as i32
                }
            })
            .collect::<Vec<_>>();
        let mut backing_dimensions = expected_dimensions.clone();
        let final_dimension_index = backing_dimensions.len() - 1;
        backing_dimensions[final_dimension_index] = backing_dimensions[final_dimension_index]
            .checked_add(1)
            .expect("the test backing dimension should fit");
        let noncontiguous_backing = runtime
            .zeros(&backing_dimensions, tensor_layout.dtype().mlx_dtype())
            .expect("the test should create noncontiguous backing storage");
        let mut slice_starts = vec![0_i32; backing_dimensions.len()];
        slice_starts[final_dimension_index] = 1;
        let slice_ends = backing_dimensions.clone();
        let slice_strides = vec![1_i32; backing_dimensions.len()];
        let noncontiguous_sequence_tensor = runtime
            .slice(
                &noncontiguous_backing,
                &slice_starts,
                &slice_ends,
                &slice_strides,
            )
            .expect("the test should create a contract-shaped noncontiguous tensor");
        assert_eq!(noncontiguous_sequence_tensor.shape(), expected_dimensions);
        let publication_workspace_bytes = noncontiguous_sequence_tensor.byte_count();
        sequence_state_tensors.insert(
            sequence_tensor_layout.persistent_tensor_name(),
            noncontiguous_sequence_tensor,
        );
        let artificial_active_pressure = runtime
            .zeros(&[8_000_000], MlxDtype::BFloat16)
            .expect("the test should create artificial active pressure");
        let captured_arrays = sequence_state_tensors
            .values()
            .chain(boundary_state_tensors.values())
            .chain([&noncontiguous_backing, &artificial_active_pressure])
            .collect::<Vec<_>>();
        runtime
            .evaluate_arrays(&captured_arrays)
            .expect("captured arrays and artificial pressure should materialize");
        drop(captured_arrays);
        let active_memory_bytes_with_pressure = runtime
            .memory_snapshot()
            .expect("the test should observe active pressure")
            .active_memory_bytes();
        let pressure_release_credit_bytes = artificial_active_pressure.byte_count() / 2;
        let constrained_active_memory_limit_bytes = active_memory_bytes_with_pressure
            .saturating_add(publication_workspace_bytes)
            .saturating_sub(pressure_release_credit_bytes);
        let constrained_memory_limits = MlxMemoryLimits::new(
            constrained_active_memory_limit_bytes,
            original_memory_limits.allocator_cache_memory_limit_bytes(),
        )
        .expect("the constrained publication limits should be valid");
        let mut memory_limit_guard = RuntimeMemoryLimitGuard::new(
            runtime,
            original_memory_limits,
            constrained_memory_limits,
        );

        let first_publication_error = persistent_prompt_cache
            .publish_block(
                memory_limit_guard.runtime(),
                &block_key,
                None,
                &sequence_state_tensors,
                &boundary_state_tensors,
            )
            .expect_err("artificial active ownership should reject the first direct write");
        assert!(
            first_publication_error
                .active_memory_deficit_bytes()
                .is_some()
        );
        assert_eq!(persistent_prompt_cache.sequence_state_block_count(), 0);
        assert_eq!(persistent_prompt_cache.boundary_state_snapshot_count(), 0);

        drop(artificial_active_pressure);
        memory_limit_guard
            .runtime()
            .synchronize_gpu_stream_and_clear_allocator_cache()
            .expect("releasing artificial ownership should reclaim publication capacity");
        let retry_outcome = persistent_prompt_cache
            .publish_block(
                memory_limit_guard.runtime(),
                &block_key,
                None,
                &sequence_state_tensors,
                &boundary_state_tensors,
            )
            .expect("the unchanged captured arrays should publish on the one retry");
        assert_eq!(
            retry_outcome,
            PersistentPromptCachePublicationOutcome::Published
        );
        memory_limit_guard.restore();

        let loaded_sequence_state = persistent_prompt_cache
            .load_kv_block(memory_limit_guard.runtime(), &block_key, None)
            .expect("the retried sequence state should load")
            .expect("the retried sequence state should be present");
        let loaded_boundary_state = persistent_prompt_cache
            .load_recurrent_snapshot(memory_limit_guard.runtime(), &block_key, None)
            .expect("the retried boundary state should load")
            .expect("the retried boundary state should be present");
        super::persistent_prompt_cache_disk_store_support::assert_split_tensor_shapes_match(
            &loaded_sequence_state,
            &sequence_state_tensors,
        );
        super::persistent_prompt_cache_disk_store_support::assert_split_tensor_shapes_match(
            &loaded_boundary_state,
            &boundary_state_tensors,
        );
    })
    .await
    .expect("the constrained direct-publication retry should finish within 115 seconds");
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
            .expect("the test should install constrained publication limits");
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
