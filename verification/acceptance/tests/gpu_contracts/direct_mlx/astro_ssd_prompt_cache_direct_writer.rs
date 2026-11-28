use std::collections::HashMap;
use std::time::Duration;

use model_serving::{
    AstroSsdPromptCacheBlockKey, AstroSsdPromptCacheDiskStore,
    AstroSsdPromptCachePublicationOutcome, PerformanceAttribution, PerformanceOperation,
};
use runtime_integration::{MlxMemoryLimits, MlxRuntime};

use super::astro_ssd_prompt_cache_disk_store_support::{
    astro_ssd_prompt_cache_block_key_for_seed, block_tokens_for_seed,
    open_astro_ssd_prompt_cache_disk_store, runtime_with_shared_limits, synthetic_kv_block_tensors,
    synthetic_recurrent_snapshot_tensors,
};
use crate::common::qwen3_5_moe;
use mlx_bridge::{MlxArray, MlxDtype};

const LARGE_ASTRO_SSD_PROMPT_CACHE_LIMIT_BYTES: u64 = 10 * 1024 * 1024 * 1024;
const DIRECT_MLX_ASTRO_SSD_PROMPT_CACHE_TEST_TIMEOUT: Duration = Duration::from_secs(115);

#[tokio::test]
async fn should_publish_required_capture_from_host_copies_with_every_stage_attributed() {
    tokio::time::timeout(DIRECT_MLX_ASTRO_SSD_PROMPT_CACHE_TEST_TIMEOUT, async {
        let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
        let runtime = runtime_with_shared_limits();
        let astro_ssd_prompt_cache_directory = tempfile::tempdir()
            .expect("the test should create an astro SSD prompt cache directory");
        let astro_ssd_prompt_cache = open_astro_ssd_prompt_cache_disk_store(
            &astro_ssd_prompt_cache_directory,
            LARGE_ASTRO_SSD_PROMPT_CACHE_LIMIT_BYTES,
        )
        .expect("the astro SSD prompt cache should open");
        let sequence_state_tensors = synthetic_kv_block_tensors(&runtime);
        let boundary_state_tensors = synthetic_recurrent_snapshot_tensors(&runtime);
        let mut performance_attribution = PerformanceAttribution::enabled();

        let publication_outcome = astro_ssd_prompt_cache
            .publish_block_with_performance_attribution(
                &runtime,
                &astro_ssd_prompt_cache_block_key_for_seed(0),
                None,
                &sequence_state_tensors,
                &boundary_state_tensors,
                &mut performance_attribution,
            )
            .expect("the required capture should publish directly");

        assert_eq!(
            publication_outcome,
            AstroSsdPromptCachePublicationOutcome::Published
        );
        assert!(
            performance_attribution
                .operation_measurement(
                    PerformanceOperation::AstroSsdPromptCacheKvBlockSerialization,
                )
                .is_some(),
            "direct publication should retain write attribution"
        );
        assert!(
            performance_attribution
                .operation_measurement(
                    PerformanceOperation::AstroSsdPromptCachePublicationSynchronizationWait,
                )
                .is_some(),
            "direct publication should attribute durable synchronization"
        );
        assert!(
            performance_attribution
                .operation_measurement(
                    PerformanceOperation::AstroSsdPromptCachePublicationValidation,
                )
                .is_some(),
            "direct publication should attribute contract validation"
        );
        assert!(
            performance_attribution
                .operation_measurement(
                    PerformanceOperation::AstroSsdPromptCacheGlobalQuotaEviction,
                )
                .is_some(),
            "direct publication should attribute global quota eviction work"
        );
        assert!(
            performance_attribution
                .operation_measurement(PerformanceOperation::AstroSsdPromptCacheAtomicCommit)
                .is_some(),
            "direct publication should attribute its atomic commit"
        );
    })
    .await
    .expect("the direct publication test should finish within 115 seconds");
}

#[tokio::test]
async fn should_publish_four_production_sized_boundaries_sequentially() {
    tokio::time::timeout(DIRECT_MLX_ASTRO_SSD_PROMPT_CACHE_TEST_TIMEOUT, async {
        let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
        let runtime = runtime_with_shared_limits();
        let astro_ssd_prompt_cache_directory = tempfile::tempdir()
            .expect("the test should create an astro SSD prompt cache directory");
        let astro_ssd_prompt_cache = open_astro_ssd_prompt_cache_disk_store(
            &astro_ssd_prompt_cache_directory,
            LARGE_ASTRO_SSD_PROMPT_CACHE_LIMIT_BYTES,
        )
        .expect("the astro SSD prompt cache should open");
        let sequence_state_tensors = synthetic_kv_block_tensors(&runtime);
        let boundary_state_tensors = synthetic_recurrent_snapshot_tensors(&runtime);
        let mut parent_block_key: Option<AstroSsdPromptCacheBlockKey> = None;

        for boundary_index in 0_u32..4 {
            let block_key = match parent_block_key.as_ref() {
                None => astro_ssd_prompt_cache_block_key_for_seed(0),
                Some(parent_block_key) => parent_block_key
                    .for_child_block(&block_tokens_for_seed(boundary_index * 10_000))
                    .expect("the test should construct the next child block identity"),
            };
            let publication_outcome = astro_ssd_prompt_cache
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
                AstroSsdPromptCachePublicationOutcome::Published
            );
            parent_block_key = Some(block_key);
        }

        assert_eq!(astro_ssd_prompt_cache.sequence_state_block_count(), 4);
    })
    .await
    .expect("the sequential direct publication test should finish within 115 seconds");
}

#[tokio::test]
async fn should_retry_the_same_captured_arrays_after_releasing_artificial_active_pressure() {
    tokio::time::timeout(DIRECT_MLX_ASTRO_SSD_PROMPT_CACHE_TEST_TIMEOUT, async {
        let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
        let runtime = runtime_with_shared_limits();
        let original_memory_limits = runtime.memory_limits();
        let astro_ssd_prompt_cache_directory = tempfile::tempdir()
            .expect("the test should create an astro SSD prompt cache directory");
        let astro_ssd_prompt_cache = open_astro_ssd_prompt_cache_disk_store(
            &astro_ssd_prompt_cache_directory,
            LARGE_ASTRO_SSD_PROMPT_CACHE_LIMIT_BYTES,
        )
        .expect("the astro SSD prompt cache should open");
        let block_key = astro_ssd_prompt_cache_block_key_for_seed(0);
        let mut sequence_state_tensors = synthetic_kv_block_tensors(&runtime);
        let boundary_state_tensors = synthetic_recurrent_snapshot_tensors(&runtime);
        let noncontiguous_insertion =
            insert_noncontiguous_sequence_tensor(&runtime, &mut sequence_state_tensors);
        let artificial_active_pressure = runtime
            .zeros(&[8_000_000], MlxDtype::BFloat16)
            .expect("the test should create artificial active pressure");
        let constrained_memory_limits = constrained_publication_memory_limits(
            &runtime,
            original_memory_limits,
            &CapturedPublicationArrays {
                sequence_state_tensors: &sequence_state_tensors,
                boundary_state_tensors: &boundary_state_tensors,
                noncontiguous_insertion: &noncontiguous_insertion,
                artificial_active_pressure: &artificial_active_pressure,
            },
        );
        let mut memory_limit_guard = RuntimeMemoryLimitGuard::new(
            runtime,
            original_memory_limits,
            constrained_memory_limits,
        );

        assert_first_direct_write_is_rejected_by_active_pressure(
            memory_limit_guard.runtime(),
            &astro_ssd_prompt_cache,
            &block_key,
            &PublicationStateTensors {
                sequence_state_tensors: &sequence_state_tensors,
                boundary_state_tensors: &boundary_state_tensors,
            },
        );

        drop(artificial_active_pressure);
        memory_limit_guard
            .runtime()
            .synchronize_gpu_stream_and_clear_mlx_buffer_cache()
            .expect("releasing artificial ownership should reclaim publication capacity");
        let retry_outcome = astro_ssd_prompt_cache
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
            AstroSsdPromptCachePublicationOutcome::Published
        );
        memory_limit_guard.restore();

        assert_retried_blocks_load_with_published_shapes(
            memory_limit_guard.runtime(),
            &astro_ssd_prompt_cache,
            &block_key,
            &PublicationStateTensors {
                sequence_state_tensors: &sequence_state_tensors,
                boundary_state_tensors: &boundary_state_tensors,
            },
        );
    })
    .await
    .expect("the constrained direct-publication retry should finish within 115 seconds");
}

/// What inserting the noncontiguous sequence tensor leaves the test to keep:
/// the backing array the tensor is sliced from (one element wider on the last
/// axis, so the slice is not contiguous) and the tensor's byte count.
struct NoncontiguousSequenceInsertion {
    backing: MlxArray,
    publication_workspace_bytes: usize,
}

/// The two tensor sets one block publication writes, passed together so a
/// function that takes both stays under the argument limit without a tuple.
struct PublicationStateTensors<'a> {
    sequence_state_tensors: &'a HashMap<String, MlxArray>,
    boundary_state_tensors: &'a HashMap<String, MlxArray>,
}

/// The arrays a publication holds on to, which the pressure limits are
/// computed from once they are all materialized.
struct CapturedPublicationArrays<'a> {
    sequence_state_tensors: &'a HashMap<String, MlxArray>,
    boundary_state_tensors: &'a HashMap<String, MlxArray>,
    noncontiguous_insertion: &'a NoncontiguousSequenceInsertion,
    artificial_active_pressure: &'a MlxArray,
}

fn insert_noncontiguous_sequence_tensor(
    runtime: &MlxRuntime,
    sequence_state_tensors: &mut HashMap<String, MlxArray>,
) -> NoncontiguousSequenceInsertion {
    let model_contract = qwen3_5_moe::astro_ssd_prompt_cache_model_contract();
    let sequence_tensor_layout = model_contract
        .astro_decoder_state_cache_layout()
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
    NoncontiguousSequenceInsertion {
        backing: noncontiguous_backing,
        publication_workspace_bytes,
    }
}

fn constrained_publication_memory_limits(
    runtime: &MlxRuntime,
    original_memory_limits: MlxMemoryLimits,
    captured_arrays: &CapturedPublicationArrays<'_>,
) -> MlxMemoryLimits {
    let materialized_arrays = captured_arrays
        .sequence_state_tensors
        .values()
        .chain(captured_arrays.boundary_state_tensors.values())
        .chain([
            &captured_arrays.noncontiguous_insertion.backing,
            captured_arrays.artificial_active_pressure,
        ])
        .collect::<Vec<_>>();
    runtime
        .evaluate_arrays(&materialized_arrays)
        .expect("captured arrays and artificial pressure should materialize");
    drop(materialized_arrays);
    let active_memory_bytes_with_pressure = runtime
        .memory_snapshot()
        .expect("the test should observe active pressure")
        .active_memory_bytes();
    let pressure_release_credit_bytes = captured_arrays.artificial_active_pressure.byte_count() / 2;
    let constrained_active_memory_limit_bytes = active_memory_bytes_with_pressure
        .saturating_add(
            captured_arrays
                .noncontiguous_insertion
                .publication_workspace_bytes,
        )
        .saturating_sub(pressure_release_credit_bytes);
    MlxMemoryLimits::new(
        constrained_active_memory_limit_bytes,
        original_memory_limits.mlx_buffer_cache_limit_bytes(),
    )
    .expect("the constrained publication limits should be valid")
}

fn assert_first_direct_write_is_rejected_by_active_pressure(
    constrained_runtime: &MlxRuntime,
    astro_ssd_prompt_cache: &AstroSsdPromptCacheDiskStore,
    block_key: &AstroSsdPromptCacheBlockKey,
    captured_state_tensors: &PublicationStateTensors<'_>,
) {
    let first_publication_error = astro_ssd_prompt_cache
        .publish_block(
            constrained_runtime,
            block_key,
            None,
            captured_state_tensors.sequence_state_tensors,
            captured_state_tensors.boundary_state_tensors,
        )
        .expect_err("artificial active ownership should reject the first direct write");
    assert!(
        first_publication_error
            .active_memory_deficit_bytes()
            .is_some()
    );
    assert_eq!(astro_ssd_prompt_cache.sequence_state_block_count(), 0);
    assert_eq!(astro_ssd_prompt_cache.boundary_state_snapshot_count(), 0);
}

fn assert_retried_blocks_load_with_published_shapes(
    restored_runtime: &MlxRuntime,
    astro_ssd_prompt_cache: &AstroSsdPromptCacheDiskStore,
    block_key: &AstroSsdPromptCacheBlockKey,
    published_state_tensors: &PublicationStateTensors<'_>,
) {
    let loaded_block_state =
        super::astro_ssd_prompt_cache_disk_store_support::load_published_block_state(
            restored_runtime,
            astro_ssd_prompt_cache,
            block_key,
        );
    super::astro_ssd_prompt_cache_disk_store_support::assert_split_tensor_shapes_match(
        &loaded_block_state.sequence_state_tensors,
        published_state_tensors.sequence_state_tensors,
    );
    super::astro_ssd_prompt_cache_disk_store_support::assert_split_tensor_shapes_match(
        &loaded_block_state.boundary_state_tensors,
        published_state_tensors.boundary_state_tensors,
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
            .synchronize_gpu_stream_and_clear_mlx_buffer_cache();
    }
}
