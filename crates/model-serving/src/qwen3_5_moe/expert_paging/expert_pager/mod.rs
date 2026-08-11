//! Expert pager coordination for startup plans and routed expert loading.

mod native_cache;

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
}

/// Startup-validated sparse-expert page plans and live memory budget.
///
/// Layer plans describe immutable artifact geometry, `memory_budget` projects
/// the worker's current MLX ceiling, and `native_expert_cache` is the sole owner
/// of mutable residency and layer-balanced least-recently-used policy. No Rust page
/// cache or alternate bypass path exists beside it.
#[derive(Debug)]
pub struct Qwen3_5ExpertPager {
    pub(super) layer_plans: Vec<QuantizedExpertLayerPlan>,
    pub(super) memory_budget: LiveMetalBudget,
    pub(super) native_expert_cache: MlxNativeExpertCache,
}

impl Qwen3_5ExpertPager {
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
