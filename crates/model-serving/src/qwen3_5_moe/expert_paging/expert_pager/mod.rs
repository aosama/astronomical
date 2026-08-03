//! Expert pager coordination for startup plans and routed expert loading.

mod cache_bypass;
mod direct_page;
mod memory_cache;

use thiserror::Error;

use super::super::model::decoder_layer_weights::Qwen3_5MoEAffineWeights;
use super::memory_budget::{LiveMetalBudget, MemoryBudgetError, MemoryBudgetSnapshot};
use super::quantized_expert_manifest::{ExpertManifestError, QuantizedExpertLayerPlan};
use super::safetensors_header::SafetensorsHeaderError;

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
#[derive(Debug)]
pub struct ExpertPager {
    pub(super) layer_plans: Vec<QuantizedExpertLayerPlan>,
    pub(super) decoder_layer_plan_count: usize,
    pub(super) aligned_expert_pack_layers:
        Vec<Option<super::aligned_expert_pack_loader::AlignedExpertPackLayer>>,
    pub(super) memory_budget: LiveMetalBudget,
}

/// Quantized affine weights loaded for selected experts in one layer.
#[derive(Debug)]
pub struct PagedExpertWeights {
    pub(crate) gate_projection: Qwen3_5MoEAffineWeights,
    pub(crate) up_projection: Qwen3_5MoEAffineWeights,
    pub(crate) down_projection: Qwen3_5MoEAffineWeights,
    pub(super) _metal_expert_pack_load_owner:
        Option<astronomical_runtime_integration::MlxMetalExpertPackLoad>,
}

impl PagedExpertWeights {
    pub(crate) fn append_array_references<'weights>(
        &'weights self,
        expert_weight_arrays: &mut Vec<&'weights astronomical_runtime_integration::MlxArray>,
    ) {
        self.gate_projection
            .append_array_references(expert_weight_arrays);
        self.up_projection
            .append_array_references(expert_weight_arrays);
        self.down_projection
            .append_array_references(expert_weight_arrays);
    }
}

impl ExpertPager {
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
