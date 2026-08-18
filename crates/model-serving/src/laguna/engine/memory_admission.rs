//! Request-level Laguna context admission before decoder or cache arrays allocate.

use crate::laguna::laguna_decoder_cache_layout;
use crate::laguna::{LagunaDecoderState, LagunaModel};
use crate::{
    ContextAdmissionRequirements, InferenceEngineError, MemoryAdmissionDecision,
    PerformanceAttribution, PerformanceOperation,
};
use astronomical_runtime_integration::MlxRuntime;

use super::execution::LagunaInferenceExecution;

impl LagunaInferenceExecution {
    /// Reassesses after each ownership mutation so admission never trusts stale counters.
    pub(super) fn admit_generation_context(
        &mut self,
        context_growth_token_count: usize,
        maximum_forward_token_count: usize,
        include_boundary_growth: bool,
        restored_prompt_prefix_token_count: usize,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(), InferenceEngineError> {
        let (runtime, model) = match (self.runtime.as_ref(), self.model.as_mut()) {
            (Some(runtime), Some(model)) => (runtime, model),
            _ => {
                return Err(InferenceEngineError::Fatal {
                    reason: "Laguna context admission requires a loaded runtime and model"
                        .to_owned(),
                });
            }
        };
        admit_generation_context(
            runtime,
            model,
            None,
            context_growth_token_count,
            maximum_forward_token_count,
            include_boundary_growth,
            restored_prompt_prefix_token_count,
            performance_attribution,
        )
    }
}

pub(super) fn admit_generation_context(
    runtime: &MlxRuntime,
    model: &mut LagunaModel,
    decoder_state: Option<&LagunaDecoderState>,
    context_growth_token_count: usize,
    maximum_forward_token_count: usize,
    include_boundary_growth: bool,
    restored_prompt_prefix_token_count: usize,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<(), InferenceEngineError> {
    for _admission_attempt in 0..3 {
        let decoder_cache_layout =
            laguna_decoder_cache_layout(model.contract()).map_err(|layout_error| {
                InferenceEngineError::InvalidRequest {
                    reason: format!("Laguna context geometry is invalid: {layout_error}"),
                }
            })?;
        let decoder_memory_projection = match decoder_state {
            Some(decoder_state) => decoder_state.projected_context_admission_memory(
                model.contract(),
                context_growth_token_count,
                maximum_forward_token_count,
            ),
            None => LagunaDecoderState::empty(model.contract()).and_then(|decoder_state| {
                decoder_state.projected_context_admission_memory(
                    model.contract(),
                    context_growth_token_count,
                    maximum_forward_token_count,
                )
            }),
        }
        .map_err(|projection_error| InferenceEngineError::InvalidRequest {
            reason: format!("Laguna context growth projection is invalid: {projection_error}"),
        })?;
        let context_growth_bytes = if include_boundary_growth {
            decoder_memory_projection
                .persistent_growth_bytes()
                .checked_add(
                    decoder_cache_layout
                        .boundary_snapshot_payload_byte_count()
                        .map_err(|layout_error| InferenceEngineError::InvalidRequest {
                            reason: format!("Laguna boundary geometry is invalid: {layout_error}"),
                        })?,
                )
        } else {
            Some(decoder_memory_projection.persistent_growth_bytes())
        }
        .ok_or(InferenceEngineError::InvalidRequest {
            reason: "Laguna context memory projection overflowed".to_owned(),
        })?;
        let prompt_cache_restore_workspace_bytes = bounded_prompt_cache_restore_workspace_bytes(
            &decoder_cache_layout,
            restored_prompt_prefix_token_count,
        )?;
        let temporary_workspace_bytes = decoder_memory_projection
            .sliding_temporary_workspace_bytes()
            .checked_add(prompt_cache_restore_workspace_bytes)
            .ok_or(InferenceEngineError::InvalidRequest {
                reason: "Laguna context temporary workspace projection overflowed".to_owned(),
            })?;
        let memory_snapshot = performance_attribution
            .measure_operation(PerformanceOperation::MemoryAdmissionSnapshot, |_| {
                runtime.memory_snapshot()
            })
            .map_err(|memory_error| InferenceEngineError::Fatal {
                reason: format!("Laguna context admission could not sample memory: {memory_error}"),
            })?;
        let expert_statistics = model.expert_weight_memory_cache_statistics();
        let complete_experts_are_resident = model.native_routed_experts_are_resident();
        let expert_page_reservation_bytes = if complete_experts_are_resident {
            0
        } else {
            usize::try_from(model.maximum_expert_page_bytes()).unwrap_or(usize::MAX)
        };
        let decision = ContextAdmissionRequirements {
            current_active_memory_bytes: memory_snapshot.active_memory_bytes(),
            context_growth_bytes,
            expert_page_reservation_bytes,
            temporary_workspace_bytes,
            retained_expert_payload_bytes: usize::try_from(
                expert_statistics.resident_payload_byte_count,
            )
            .unwrap_or(usize::MAX),
            active_memory_ceiling_bytes: runtime.memory_limits().active_memory_limit_bytes(),
            complete_experts_are_resident,
        }
        .decide();
        tracing::debug!(
            context_growth_token_count,
            maximum_forward_token_count,
            include_boundary_growth,
            context_growth_bytes,
            expert_page_reservation_bytes,
            prompt_cache_restore_workspace_bytes,
            sliding_temporary_workspace_bytes =
                decoder_memory_projection.sliding_temporary_workspace_bytes(),
            ?decision,
            "Laguna applied centralized generation-context admission"
        );
        match decision {
            MemoryAdmissionDecision::Admit => return Ok(()),
            MemoryAdmissionDecision::DemoteCompleteResidency { .. } => {
                model
                    .demote_native_routed_experts(runtime, performance_attribution)
                    .map_err(|demotion_error| InferenceEngineError::Fatal {
                        reason: format!(
                            "Laguna context admission demotion failed: {demotion_error}"
                        ),
                    })?;
            }
            MemoryAdmissionDecision::Reclaim { required_bytes } => {
                model.reclaim_retained_experts_for_request_pressure(required_bytes);
                performance_attribution
                    .measure_operation(PerformanceOperation::MlxAllocatorCacheCleanup, |_| {
                        runtime.synchronize_gpu_stream_and_clear_allocator_cache()
                    })
                    .map_err(|cleanup_error| InferenceEngineError::Fatal {
                        reason: format!("Laguna context reclamation failed: {cleanup_error}"),
                    })?;
            }
            MemoryAdmissionDecision::Reject { .. } => {
                return Err(InferenceEngineError::InvalidRequest {
                    reason: "generation context exceeds available GPU wired memory".to_owned(),
                });
            }
        }
    }
    Err(InferenceEngineError::InvalidRequest {
        reason: "generation context remains above the MLX ceiling after expert reclamation"
            .to_owned(),
    })
}

/// Layer-at-a-time restore owns one source K/V pair beside the final decoder state.
fn bounded_prompt_cache_restore_workspace_bytes(
    decoder_cache_layout: &crate::DecoderCacheLayout,
    restored_prompt_prefix_token_count: usize,
) -> Result<usize, InferenceEngineError> {
    if restored_prompt_prefix_token_count == 0 {
        return Ok(0);
    }
    let sequence_layer_source_bytes = decoder_cache_layout
        .maximum_sequence_tensor_payload_byte_count(restored_prompt_prefix_token_count)
        .map_err(|layout_error| InferenceEngineError::InvalidRequest {
            reason: format!("Laguna prompt-cache sequence geometry is invalid: {layout_error}"),
        })?
        .checked_mul(2)
        .ok_or(InferenceEngineError::InvalidRequest {
            reason: "Laguna prompt-cache sequence restore workspace overflowed".to_owned(),
        })?;
    let maximum_boundary_tensor_bytes = decoder_cache_layout
        .boundary_tensor_layouts()
        .iter()
        .try_fold(0_usize, |maximum_tensor_bytes, persisted_tensor_layout| {
            persisted_tensor_layout
                .tensor_layout()
                .fixed_payload_byte_count()
                .map(|tensor_bytes| maximum_tensor_bytes.max(tensor_bytes))
        })
        .map_err(|layout_error| InferenceEngineError::InvalidRequest {
            reason: format!("Laguna prompt-cache boundary geometry is invalid: {layout_error}"),
        })?;
    let boundary_layer_source_bytes = maximum_boundary_tensor_bytes
        .checked_mul(2)
        .and_then(|paired_tensor_bytes| {
            paired_tensor_bytes.checked_add(2 * std::mem::size_of::<f32>())
        })
        .ok_or(InferenceEngineError::InvalidRequest {
            reason: "Laguna prompt-cache boundary restore workspace overflowed".to_owned(),
        })?;
    Ok(sequence_layer_source_bytes.max(boundary_layer_source_bytes))
}
