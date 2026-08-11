//! Qwen pager coordination around the native layer-balanced cache.
//!
//! Rust owns model-level memory admission and typed errors. C++ owns route
//! synchronization, page loading, layer-aware recency, eviction, and immutable page
//! snapshots. Keeping that division prevents routed tensor payloads from being
//! copied through Rust during prompt processing or decode.

use astronomical_runtime_integration::{
    MlxArray, MlxNativeExpertCacheRequestReport, MlxNativeExpertCacheSnapshot,
    MlxNativeExpertCacheStatistics, MlxRuntime,
};

use super::{ExpertPagingError, Qwen3_5ExpertPager};
use crate::expert_paging::automatic_expert_weight_memory_cache_maximum_size_bytes;

impl Qwen3_5ExpertPager {
    pub fn prepare_native_expert_snapshot(
        &self,
        runtime: &MlxRuntime,
        layer_index: usize,
        selected_expert_indices: &MlxArray,
        collect_performance_metrics: bool,
    ) -> Result<
        (
            MlxNativeExpertCacheSnapshot,
            MlxNativeExpertCacheRequestReport,
        ),
        ExpertPagingError,
    > {
        // The order is intentional:
        // 1. Ask native analysis for exact unique and missing experts.
        // 2. Sample memory after that analysis, because evaluating the lazy
        //    router may itself execute earlier graphics-processor work.
        // 3. Budget exact missing bytes plus one future routed page.
        // 4. Commit that ceiling and route together.
        //
        // If 1,000 token assignments all select the same warm expert, missing
        // bytes are zero. Treating 1,000 assignments as pages would evict useful
        // cache entries even though this route needs no new expert allocation.
        let (route_analysis, route_analysis_report) = self
            .native_expert_cache
            .analyze_layer(
                runtime,
                layer_index,
                selected_expert_indices,
                collect_performance_metrics,
            )
            .map_err(|error| ExpertPagingError::Runtime {
                description: error.to_string(),
            })?;
        let missing_route_payload_byte_count =
            route_analysis_report.missing_route_payload_byte_count();
        let memory_budget_snapshot = self.memory_budget.snapshot(
            runtime,
            "native_expert_cache_route",
            missing_route_payload_byte_count,
        )?;
        let native_expert_cache_statistics = self.native_expert_cache.statistics();
        let maximum_resident_payload_byte_count =
            automatic_expert_weight_memory_cache_maximum_size_bytes(
                &memory_budget_snapshot,
                native_expert_cache_statistics.resident_payload_byte_count(),
                missing_route_payload_byte_count,
            );
        // The returned snapshot is a separate shared owner. Later cache eviction
        // changes future routes, but cannot invalidate addresses used by this one.
        self.native_expert_cache
            .commit_layer_with_maximum_resident_payload_byte_count(
                runtime,
                route_analysis,
                maximum_resident_payload_byte_count,
                collect_performance_metrics,
            )
            .map_err(|error| ExpertPagingError::Runtime {
                description: error.to_string(),
            })
    }

    pub fn native_expert_cache_statistics(&self) -> MlxNativeExpertCacheStatistics {
        self.native_expert_cache.statistics()
    }

    pub fn update_native_expert_retention_ceiling(
        &self,
        maximum_resident_payload_byte_count: u64,
    ) -> Result<(), ExpertPagingError> {
        self.with_native_expert_cache(|native_expert_cache| {
            native_expert_cache
                .update_maximum_resident_payload_byte_count(maximum_resident_payload_byte_count)
        })
    }

    /// Prevents new retained pages while preserving the current hot set.
    pub fn freeze_native_expert_retention_growth(&self) -> bool {
        self.native_expert_cache.freeze_retention_growth()
    }

    /// Evicts at least the requested payload when possible and freezes regrowth.
    /// Existing immutable snapshots remain valid until their lazy work completes.
    pub fn reclaim_native_expert_payload_bytes(
        &self,
        reclamation_target_byte_count: u64,
    ) -> Result<bool, ExpertPagingError> {
        self.with_native_expert_cache(|native_expert_cache| {
            native_expert_cache.reclaim_retained_payload_bytes(reclamation_target_byte_count)
        })
    }

    /// Restores the configured retention ceiling for subsequent routes.
    pub fn resume_native_expert_retention_growth(&self) -> bool {
        self.native_expert_cache.resume_retention_growth()
    }

    fn with_native_expert_cache<Output>(
        &self,
        operation: impl FnOnce(
            &astronomical_runtime_integration::MlxNativeExpertCache,
        )
            -> Result<Output, astronomical_runtime_integration::MlxRuntimeError>,
    ) -> Result<Output, ExpertPagingError> {
        operation(&self.native_expert_cache).map_err(|error| ExpertPagingError::Runtime {
            description: error.to_string(),
        })
    }
}
