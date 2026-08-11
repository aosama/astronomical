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
use crate::expert_paging::{
    automatic_expert_weight_memory_cache_maximum_size_bytes,
    maximum_possible_expert_route_payload_bytes,
};

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
        let layer_plan =
            self.layer_plans
                .get(layer_index)
                .ok_or(ExpertPagingError::LayerIndexOutOfRange {
                    layer_index,
                    layer_count: self.layer_plans.len(),
                })?;
        let payload_bytes_per_expert = layer_plan.tensor_sources.iter().try_fold(
            0u64,
            |accumulated_payload_bytes, tensor_source| {
                let tensor_payload_bytes_per_expert = u64::try_from(tensor_source.bytes_per_expert)
                    .map_err(|_| ExpertPagingError::Runtime {
                        description: "expert tensor payload exceeds the portable memory range"
                            .to_owned(),
                    })?;
                accumulated_payload_bytes
                    .checked_add(tensor_payload_bytes_per_expert)
                    .ok_or_else(|| ExpertPagingError::Runtime {
                        description: "expert page payload exceeds the memory range".to_owned(),
                    })
            },
        )?;
        let maximum_possible_route_payload_bytes = maximum_possible_expert_route_payload_bytes(
            payload_bytes_per_expert,
            layer_plan.expert_capacity,
            selected_expert_indices.element_count(),
        )
        .ok_or_else(|| ExpertPagingError::Runtime {
            description: "selected expert route payload exceeds the memory range".to_owned(),
        })?;
        // Recompute retention from live MLX counters before every ordinary
        // route. Retained expert pages are the elastic category; model weights
        // and request state must not be displaced to preserve a stale cache cap.
        // Router IDs are still lazy at this boundary. Synchronizing them inside
        // native preparation can materialize the preceding decoder layer before
        // source reads begin, so reserve the complete possible distinct route
        // and evict retention before that synchronization increases residency.
        let memory_budget_snapshot = self.memory_budget.snapshot(
            runtime,
            "native_expert_cache_route",
            maximum_possible_route_payload_bytes,
        )?;
        let native_expert_cache_statistics = self.native_expert_cache.statistics();
        let maximum_resident_payload_byte_count =
            automatic_expert_weight_memory_cache_maximum_size_bytes(
                &memory_budget_snapshot,
                native_expert_cache_statistics.resident_payload_byte_count(),
                0,
            );
        // The returned snapshot retains every page that the lazy projection
        // may dereference, even if native policy evicts the corresponding cache
        // entry before MLX evaluates this graph.
        self.native_expert_cache
            .update_maximum_resident_payload_byte_count(maximum_resident_payload_byte_count)
            .map_err(|error| ExpertPagingError::Runtime {
                description: error.to_string(),
            })?;
        self.native_expert_cache
            .prepare_layer(
                runtime,
                layer_index,
                selected_expert_indices,
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
