//! Rust-owned Qwen expert streaming and resident-promotion source metadata.

mod rust_layer_streaming;

use std::fs::File;
use std::path::PathBuf;

use astronomical_runtime_integration::MlxRuntimeError;
use thiserror::Error;

use crate::expert_paging::{
    ExpertManifestError, ExpertWeightPage, QuantizedExpertLayerPlan, QuantizedExpertPageManifest,
    SafetensorsHeaderError,
};
use crate::qwen3_5::model::decoder_layer_weights::Qwen3_5AffineWeights;
use crate::{MlxAllocationBudget, MlxAllocationBudgetError};

/// Typed failures during expert streaming and resident-source handling.
#[derive(Debug, Error)]
pub enum ExpertPagingError {
    #[error("expert manifest construction failed: {0}")]
    Manifest(#[from] ExpertManifestError),
    #[error("safetensors header parsing failed: {0}")]
    SafetensorsHeader(#[from] SafetensorsHeaderError),
    #[error("memory budget exceeded: {0}")]
    MemoryBudget(#[from] MlxAllocationBudgetError),
    #[error("native MLX runtime error during expert loading: {0}")]
    NativeRuntime(#[from] MlxRuntimeError),
    #[error("invalid expert streaming plan: {description}")]
    InvalidPagingPlan { description: String },
    #[error("Rust expert streaming failed: {description}")]
    Runtime { description: String },
    #[error("layer index {layer_index} is out of range (decoder layer count: {layer_count})")]
    LayerIndexOutOfRange {
        layer_index: usize,
        layer_count: usize,
    },
    #[error("expert paging is not enabled for this model")]
    PagingNotEnabled,
    #[error("failed to open resident expert source {source_file:?}: {source}")]
    ResidentSourceOpen {
        source_file: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to clone resident expert source {source_file:?}: {source}")]
    ResidentSourceClone {
        source_file: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

/// Immutable layer geometry plus bounded-read and resident-promotion sources.
#[derive(Debug)]
pub struct Qwen3_5ExpertPager {
    pub(super) layer_plans: Vec<QuantizedExpertLayerPlan>,
    pub(super) memory_budget: MlxAllocationBudget,
    pub(super) resident_expert_source_files: Vec<(PathBuf, File)>,
}

/// Exact compact or complete expert arrays loaded by Rust for one layer use.
#[derive(Debug)]
pub struct Qwen3_5PagedExpertWeights {
    pub(crate) gate_projection: Qwen3_5AffineWeights,
    pub(crate) up_projection: Qwen3_5AffineWeights,
    pub(crate) down_projection: Qwen3_5AffineWeights,
}

impl Qwen3_5PagedExpertWeights {
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

impl ExpertWeightPage for Qwen3_5PagedExpertWeights {
    fn resident_payload_byte_count(&self) -> u64 {
        affine_payload_byte_count(&self.gate_projection)
            .saturating_add(affine_payload_byte_count(&self.up_projection))
            .saturating_add(affine_payload_byte_count(&self.down_projection))
    }
}

fn affine_payload_byte_count(affine_weights: &Qwen3_5AffineWeights) -> u64 {
    let mut arrays = Vec::new();
    affine_weights.append_array_references(&mut arrays);
    arrays.into_iter().fold(0u64, |payload_bytes, array| {
        payload_bytes.saturating_add(u64::try_from(array.byte_count()).unwrap_or(u64::MAX))
    })
}

/// One complete Rust-loaded layer retained between requests.
#[derive(Debug)]
pub(crate) struct Qwen3_5RetainedExpertLayer {
    pub(crate) weights: Qwen3_5PagedExpertWeights,
    pub(crate) manifest: QuantizedExpertPageManifest,
}

impl Qwen3_5RetainedExpertLayer {
    pub(crate) fn has_exact_expert_ids(&self, expert_ids: &[usize]) -> bool {
        self.manifest.expert_ids == expert_ids
    }

    pub(crate) fn contains_every_expert(&self, selected_expert_ids: &[usize]) -> bool {
        self.manifest.contains_every_expert(selected_expert_ids)
    }
}

impl ExpertWeightPage for Qwen3_5RetainedExpertLayer {
    fn resident_payload_byte_count(&self) -> u64 {
        self.weights.resident_payload_byte_count()
    }
}

impl Qwen3_5ExpertPager {
    pub(crate) fn layer_plans(&self) -> &[QuantizedExpertLayerPlan] {
        &self.layer_plans
    }

    pub(crate) fn complete_expert_payload_byte_count(&self) -> Result<u64, ExpertPagingError> {
        self.layer_plans
            .iter()
            .try_fold(0_u64, |model_payload_bytes, layer_plan| {
                model_payload_bytes
                    .checked_add(layer_plan.complete_expert_payload_byte_count()?)
                    .ok_or_else(|| ExpertPagingError::InvalidPagingPlan {
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

    pub(crate) fn remove_resident_expert_source_files_for_tests(&mut self) {
        self.resident_expert_source_files.clear();
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
            .update_active_memory_ceiling_bytes(configured_mlx_memory_ceiling_bytes);
    }

    /// Rust-streamed pages are operation-local, so adaptive admission has no
    /// retained streaming owner to re-plan before the next forward.
    pub(crate) fn set_pending_admitted_forward_reserve_bytes(
        &self,
        _admitted_forward_reserve_bytes: u64,
    ) {
    }

    pub(crate) fn update_observed_transient_high_water_bytes(
        &self,
        observed_transient_high_water_bytes: u64,
    ) {
        self.memory_budget
            .update_observed_transient_high_water_bytes(observed_transient_high_water_bytes);
    }

    #[must_use]
    pub(crate) fn observed_transient_high_water_bytes(&self) -> u64 {
        self.memory_budget.observed_transient_high_water_bytes()
    }

    pub(crate) fn configured_mlx_memory_ceiling_bytes(&self) -> u64 {
        self.memory_budget.active_memory_ceiling_bytes()
    }

    pub(crate) fn maximum_expert_page_bytes(&self) -> u64 {
        self.memory_budget.maximum_expert_page_bytes()
    }

    /// Returns the largest exact page payload for one routed top-K set.
    pub(crate) fn maximum_routed_expert_page_bytes(
        &self,
        experts_per_token: usize,
    ) -> Result<u64, ExpertPagingError> {
        self.layer_plans
            .iter()
            .try_fold(0_u64, |largest_page_bytes, layer_plan| {
                let complete_layer_payload_bytes =
                    layer_plan.complete_expert_payload_byte_count()?;
                let routed_expert_count = experts_per_token.min(layer_plan.expert_capacity);
                let routed_page_bytes = u128::from(complete_layer_payload_bytes)
                    .saturating_mul(routed_expert_count as u128)
                    / (layer_plan.expert_capacity.max(1) as u128);
                Ok(largest_page_bytes.max(u64::try_from(routed_page_bytes).unwrap_or(u64::MAX)))
            })
    }
}
