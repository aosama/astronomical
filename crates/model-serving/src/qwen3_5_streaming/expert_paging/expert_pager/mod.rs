//! Rust-owned Qwen expert streaming and resident-promotion source metadata.

mod rust_expert_streaming;

pub(crate) use rust_expert_streaming::Qwen3_5ExpertStreamingRequestShape;

use std::cell::Cell;
use std::fs::File;
use std::mem::size_of;
use std::path::PathBuf;

use astronomical_runtime_integration::MlxRuntimeError;
use thiserror::Error;

use crate::expert_paging::{
    ExpertManifestError, ExpertWeightPage, QuantizedExpertLayerPlan, SafetensorsHeaderError,
};
use crate::qwen3_5_core::model_math::decoder_layer_weights::Qwen3_5AffineWeights;
use crate::{MlxAllocationAdmission, MlxAllocationAdmissionError};
use astronomical_mlx_c_rust::MlxArray;

/// Typed failures during expert streaming and resident-source handling.
#[derive(Debug, Error)]
pub enum ExpertPagingError {
    #[error("expert manifest construction failed: {0}")]
    Manifest(#[from] ExpertManifestError),
    #[error("safetensors header parsing failed: {0}")]
    SafetensorsHeader(#[from] SafetensorsHeaderError),
    #[error("memory budget exceeded: {0}")]
    MemoryBudget(#[from] MlxAllocationAdmissionError),
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
}

/// Translates streaming-side paging failures into the family execution error.
///
/// The paging error type stays owned by this module; the family error lives in
/// the shared kernel and carries only crate-level memory-admission evidence,
/// so the kernel never depends on streaming types (issue #1132). Memory-budget
/// rejections surface as the kernel's admission variant; native runtime
/// failures surface as the kernel's runtime variant.
impl From<ExpertPagingError> for crate::qwen3_5_core::model_math::error::Qwen3_5ExecutionError {
    fn from(expert_paging_error: ExpertPagingError) -> Self {
        match expert_paging_error {
            ExpertPagingError::MemoryBudget(memory_admission_error) => {
                Self::MemoryAdmissionRejected(memory_admission_error)
            }
            ExpertPagingError::NativeRuntime(mlx_runtime_error) => Self::Runtime(mlx_runtime_error),
            remaining_expert_paging_error => Self::ExpertSourceFailure {
                description: remaining_expert_paging_error.to_string(),
            },
        }
    }
}

/// Immutable layer geometry plus bounded-read and resident-promotion sources.
#[derive(Debug)]
pub struct Qwen3_5ExpertPager {
    pub(super) layer_plans: Vec<QuantizedExpertLayerPlan>,
    pub(super) memory_budget: MlxAllocationAdmission,
    pub(super) resident_expert_source_files: Vec<(PathBuf, File)>,
    /// True once the model owns the fused jangtq2 gathered projection kernel:
    /// single-token pages then multiply straight from their packed codes, so
    /// their admission and reservation price only the packed source bytes.
    /// Multi-token pages still decode a transient bfloat16 copy and keep the
    /// full decode peak. The model sets this after the capability verdict.
    pub(super) decode_tokens_multiply_packed_pages: Cell<bool>,
}

/// Exact compact or complete expert arrays loaded by Rust for one layer use.
#[derive(Debug)]
pub struct Qwen3_5PagedExpertWeights {
    pub(crate) gate_projection: Qwen3_5AffineWeights,
    pub(crate) up_projection: Qwen3_5AffineWeights,
    pub(crate) down_projection: Qwen3_5AffineWeights,
    pub(crate) gate_activation_rotation:
        crate::jangtq2_qwen35::configuration::Jangtq2CodebookRotation,
    pub(crate) up_activation_rotation:
        crate::jangtq2_qwen35::configuration::Jangtq2CodebookRotation,
    pub(crate) down_activation_rotation:
        crate::jangtq2_qwen35::configuration::Jangtq2CodebookRotation,
}

impl ExpertWeightPage for Qwen3_5PagedExpertWeights {
    fn resident_payload_byte_count(&self) -> u64 {
        affine_payload_byte_count(&self.gate_projection)
            .saturating_add(affine_payload_byte_count(&self.up_projection))
            .saturating_add(affine_payload_byte_count(&self.down_projection))
    }
}

impl Qwen3_5PagedExpertWeights {
    pub(crate) fn append_array_references<'weights>(
        &'weights self,
        arrays: &mut Vec<&'weights MlxArray>,
    ) {
        self.gate_projection.append_array_references(arrays);
        self.up_projection.append_array_references(arrays);
        self.down_projection.append_array_references(arrays);
    }
}

fn affine_payload_byte_count(affine_weights: &Qwen3_5AffineWeights) -> u64 {
    let mut arrays = Vec::new();
    affine_weights.append_array_references(&mut arrays);
    arrays.into_iter().fold(0u64, |payload_bytes, array| {
        payload_bytes.saturating_add(u64::try_from(array.byte_count()).unwrap_or(u64::MAX))
    })
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

    pub(crate) fn update_observed_transient_high_water_bytes(
        &self,
        observed_transient_high_water_bytes: u64,
    ) {
        self.memory_budget
            .update_observed_transient_high_water_bytes(observed_transient_high_water_bytes);
    }

    pub(crate) fn configured_mlx_memory_ceiling_bytes(&self) -> u64 {
        self.memory_budget.active_memory_ceiling_bytes()
    }

    pub(crate) fn maximum_expert_page_bytes(&self) -> u64 {
        self.memory_budget.maximum_expert_page_bytes()
    }

    /// Largest complete-layer packed source payload across the model's layers.
    ///
    /// Sizes the RAM owner's complete-layer streaming slot from on-disk source
    /// geometry only. Decoded bfloat16 output is reserved separately through
    /// the expert-page reservation, so the two roles cannot inflate each
    /// other's budgets.
    pub(crate) fn maximum_complete_expert_source_payload_bytes(
        &self,
    ) -> Result<u64, ExpertPagingError> {
        self.layer_plans
            .iter()
            .try_fold(0_u64, |largest_source_layer_bytes, layer_plan| {
                Ok(
                    largest_source_layer_bytes
                        .max(layer_plan.complete_expert_payload_byte_count()?),
                )
            })
    }

    /// Records whether decode tokens multiply packed codebook pages directly.
    pub(crate) fn set_decode_tokens_multiply_packed_pages(&self, multiplies_packed_pages: bool) {
        self.decode_tokens_multiply_packed_pages
            .set(multiplies_packed_pages);
    }

    /// Returns whether a page streamed for one decode token is used packed,
    /// with no decoded output to admit.
    pub(crate) fn decode_tokens_multiply_packed_pages(&self) -> bool {
        self.decode_tokens_multiply_packed_pages.get()
    }

    /// Returns the largest peak allocation for one routed page of
    /// `experts_per_token` experts. `materializes_decoded_weights` adds the
    /// codebook decode output; a page the fused decode-token
    /// kernel multiplies in packed form costs only its source bytes.
    pub(crate) fn maximum_routed_expert_page_bytes(
        &self,
        experts_per_token: usize,
        materializes_decoded_weights: bool,
    ) -> Result<u64, ExpertPagingError> {
        self.layer_plans
            .iter()
            .try_fold(0_u64, |largest_page_peak_bytes, layer_plan| {
                let routed_expert_count = experts_per_token.min(layer_plan.expert_capacity);
                let per_expert_source_bytes = layer_plan.expert_payload_byte_count()?;
                let routed_source_page_bytes = crate::memory::source_expert_page_payload_bytes(
                    per_expert_source_bytes,
                    routed_expert_count,
                )
                .ok_or_else(|| ExpertPagingError::InvalidPagingPlan {
                    description: "routed expert source-page byte count overflowed".to_owned(),
                })?;
                if !materializes_decoded_weights {
                    return Ok(largest_page_peak_bytes.max(routed_source_page_bytes));
                }
                let decoded_codebook_weight_shapes =
                    layer_plan.jangtq2_decoded_weight_page_shapes(routed_expert_count)?;
                let routed_page_peak_bytes = crate::memory::expert_page_peak_allocation_bytes(
                    routed_source_page_bytes,
                    &decoded_codebook_weight_shapes,
                    size_of::<u16>(),
                )
                .ok_or_else(|| ExpertPagingError::InvalidPagingPlan {
                    description: "routed expert page peak allocation byte count overflowed"
                        .to_owned(),
                })?;
                Ok(largest_page_peak_bytes.max(routed_page_peak_bytes))
            })
    }
}
