//! Expert pager coordination for startup plans and routed expert loading.

mod native_cache;

use std::fs::File;
use std::path::PathBuf;

use thiserror::Error;

use astronomical_runtime_integration::MlxNativeExpertCache;

use crate::expert_paging::{
    ExpertManifestError, LiveMetalBudget, MemoryBudgetError, MemoryBudgetSnapshot,
    QuantizedExpertLayerPlan, SafetensorsHeaderError,
};

/// Typed failures during expert paging operations.
#[derive(Debug, Error)]
pub enum ExpertPagingError {
    #[error("expert manifest construction failed: {0}")]
    Manifest(#[from] ExpertManifestError),
    #[error("safetensors header parsing failed: {0}")]
    SafetensorsHeader(#[from] SafetensorsHeaderError),
    #[error("memory budget exceeded: {0}")]
    MemoryBudget(#[from] MemoryBudgetError),
    #[error("MLX runtime error during expert page loading: {description}")]
    Runtime { description: String },
    #[error("layer index {layer_index} is out of range (decoder layer count: {layer_count})")]
    LayerIndexOutOfRange {
        layer_index: usize,
        layer_count: usize,
    },
    #[error("expert paging is not enabled for this model")]
    PagingNotEnabled,
    #[error("failed to open validated expert source file {source_file:?}: {source}")]
    ResidentSourceOpen {
        source_file: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to duplicate validated expert source file {source_file:?}: {source}")]
    ResidentSourceClone {
        source_file: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

/// Startup-validated sparse-expert page plans and live memory budget.
///
/// Layer plans describe immutable artifact geometry, `memory_budget` projects
/// the worker's current MLX ceiling, and `native_expert_cache` solely owns paged
/// residency and layer-balanced least-recently-used policy. Retained source files
/// let the model construct a separate complete resident owner without reopening
/// user paths; the pager itself never stores a second Rust page cache.
#[derive(Debug)]
pub struct Qwen3_5ExpertPager {
    pub(super) layer_plans: Vec<QuantizedExpertLayerPlan>,
    pub(super) memory_budget: LiveMetalBudget,
    pub(super) native_expert_cache: MlxNativeExpertCache,
    /// One descriptor per unique validated shard, retained for future promotion.
    pub(super) resident_expert_source_files: Vec<(PathBuf, File)>,
}

impl Qwen3_5ExpertPager {
    /// Returns immutable target-then-MTP plans shared by both ownership modes.
    pub(crate) fn layer_plans(&self) -> &[QuantizedExpertLayerPlan] {
        &self.layer_plans
    }

    /// Sums the exact complete payload across every target and optional MTP layer.
    pub(crate) fn complete_expert_payload_byte_count(&self) -> Result<u64, ExpertPagingError> {
        self.layer_plans
            .iter()
            .try_fold(0_u64, |complete_model_payload_bytes, layer_plan| {
                let complete_layer_payload_bytes =
                    layer_plan.complete_expert_payload_byte_count()?;
                complete_model_payload_bytes
                    .checked_add(complete_layer_payload_bytes)
                    .ok_or_else(|| ExpertPagingError::Runtime {
                        description: "complete model expert payload byte count overflowed"
                            .to_owned(),
                    })
            })
    }

    pub(crate) fn complete_expert_entry_count(&self) -> usize {
        self.layer_plans
            .iter()
            .fold(0_usize, |entry_count, layer_plan| {
                entry_count.saturating_add(layer_plan.expert_capacity)
            })
    }

    /// Clones descriptors for one promotion attempt without consuming fallback ownership.
    pub(crate) fn clone_resident_expert_source_files(
        &self,
    ) -> Result<Vec<(PathBuf, File)>, ExpertPagingError> {
        self.resident_expert_source_files
            .iter()
            .map(|(source_file_path, source_file)| {
                source_file
                    .try_clone()
                    .map(|cloned_source_file| (source_file_path.clone(), cloned_source_file))
                    .map_err(|source| ExpertPagingError::ResidentSourceClone {
                        source_file: source_file_path.clone(),
                        source,
                    })
            })
            .collect()
    }

    /// Removes only Rust promotion descriptors for failed-transition qualification.
    pub(crate) fn remove_resident_expert_source_files_for_tests(&mut self) {
        self.resident_expert_source_files.clear();
    }

    pub(crate) fn native_expert_retention_growth_is_enabled_for_tests(&self) -> bool {
        if !self.freeze_native_expert_retention_growth() {
            return false;
        }
        self.resume_native_expert_retention_growth()
    }

    pub(crate) fn layer_plan(
        &self,
        layer_index: usize,
    ) -> Result<&QuantizedExpertLayerPlan, ExpertPagingError> {
        self.layer_plans
            .get(layer_index)
            .ok_or(ExpertPagingError::LayerIndexOutOfRange {
                layer_index,
                layer_count: self.layer_plans.len(),
            })
    }

    pub(crate) fn update_configured_mlx_memory_ceiling_bytes(
        &mut self,
        configured_mlx_memory_ceiling_bytes: u64,
    ) {
        self.memory_budget
            .update_configured_cap_bytes(configured_mlx_memory_ceiling_bytes);
    }

    pub(crate) fn configured_mlx_memory_ceiling_bytes(&self) -> u64 {
        self.memory_budget.configured_cap_bytes()
    }

    pub(crate) fn maximum_expert_page_bytes(&self) -> u64 {
        self.memory_budget.maximum_expert_page_bytes()
    }

    pub(crate) fn memory_budget_snapshot_for_mlx_memory_limit_adjustment(
        &self,
        runtime: &astronomical_runtime_integration::MlxRuntime,
    ) -> Result<MemoryBudgetSnapshot, MemoryBudgetError> {
        self.memory_budget
            .snapshot(runtime, "mlx_memory_limit_adjustment", 0)
    }
}
