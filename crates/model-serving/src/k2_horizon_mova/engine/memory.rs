//! Live MLX memory snapshots and persistent prompt-cache stats for K2.

use astronomical_ipc_protocol::WorkerEvent;

use crate::k2_horizon_mova::configuration::{K2HorizonMoVAConfig, K2HorizonMoVALayerKind};
use crate::k2_horizon_mova::k2_horizon_mova_expert_layer_geometries;
use crate::k2_horizon_mova::model::K2HorizonMoVAKvState;
use crate::k2_horizon_mova::model::K2HorizonMoVAModel;
use crate::memory::{
    BOOTSTRAP_CONTEXT_WINDOW_RESERVE_BYTES, ExpertLayerResidencyTarget, MemoryPhase,
    plan_expert_residency,
};
use crate::{
    ExpertResidencyTelemetry, InferenceEngineError, MlxActiveMemoryBreakdown, MlxMemoryTelemetry,
    PersistentPromptCacheDiskStoreConfig, build_persistent_prompt_cache_stats_event,
};

use super::execution::K2HorizonMoVAInferenceExecution;

pub(super) fn reject_unenacted_expert_streaming(
    model: &K2HorizonMoVAModel,
    mlx_memory_ceiling_bytes: u64,
) -> Result<(), InferenceEngineError> {
    let sparse_layer_payloads = model.weights.sparse_layer_expert_payloads();
    if sparse_layer_payloads.is_empty() {
        return Ok(());
    }
    let geometries = k2_horizon_mova_expert_layer_geometries(&model.config, &sparse_layer_payloads)
        .map_err(|geometry_error| InferenceEngineError::Fatal {
            reason: format!("K2 Horizon MoVA expert geometry is invalid: {geometry_error}"),
        })?;
    let retained_expert_ceiling_bytes = mlx_memory_ceiling_bytes
        .saturating_sub(model.weights.model_core_payload_bytes())
        .saturating_sub(BOOTSTRAP_CONTEXT_WINDOW_RESERVE_BYTES)
        .max(1);
    let residency_plan = plan_expert_residency(
        MemoryPhase::Prefill,
        retained_expert_ceiling_bytes,
        &geometries,
        &[],
    )
    .map_err(|plan_error| InferenceEngineError::Fatal {
        reason: format!("K2 Horizon MoVA expert residency could not be planned: {plan_error}"),
    })?;
    let requires_streaming = residency_plan.layer_targets.iter().any(|layer_target| {
        matches!(
            layer_target,
            ExpertLayerResidencyTarget::StreamOperationLocal
                | ExpertLayerResidencyTarget::AdmitPartialOnMandatoryRouteRead
        )
    });
    if requires_streaming {
        return Err(InferenceEngineError::Fatal {
            reason: format!(
                "this MLX ceiling of {mlx_memory_ceiling_bytes} bytes cannot keep every K2 Horizon MoVA expert stack resident; complete-layer streaming is not enacted yet"
            ),
        });
    }
    Ok(())
}

impl K2HorizonMoVAInferenceExecution {
    pub(super) fn persistent_prompt_cache_stats_event(&self) -> Option<WorkerEvent> {
        let persistent_prompt_cache = self.persistent_prompt_cache.as_ref()?;
        let global_prompt_cache_maximum_size_bytes = self
            .persistent_prompt_cache_disk_store_config
            .as_ref()
            .map(PersistentPromptCacheDiskStoreConfig::global_prompt_cache_maximum_size_bytes)
            .unwrap_or(0);
        Some(build_persistent_prompt_cache_stats_event(
            &self.persistent_prompt_cache_counters,
            u64::try_from(
                persistent_prompt_cache
                    .model_contract_ref()
                    .block_token_count(),
            )
            .unwrap_or(u64::MAX),
            u64::try_from(persistent_prompt_cache.sequence_state_block_count()).unwrap_or(u64::MAX),
            u64::try_from(persistent_prompt_cache.boundary_state_snapshot_count())
                .unwrap_or(u64::MAX),
            u64::try_from(persistent_prompt_cache.visual_embedding_count()).unwrap_or(u64::MAX),
            persistent_prompt_cache.total_size_bytes(),
            persistent_prompt_cache.visual_embedding_total_size_bytes(),
            global_prompt_cache_maximum_size_bytes,
        ))
    }

    pub(super) fn collect_current_mlx_memory_telemetry(&self) -> Option<MlxMemoryTelemetry> {
        let model = self.model.as_ref()?;
        let memory_snapshot = match model.runtime.memory_snapshot() {
            Ok(memory_snapshot) => memory_snapshot,
            Err(memory_snapshot_error) => {
                tracing::warn!(
                    error = %memory_snapshot_error,
                    "K2 Horizon MoVA could not sample current MLX memory"
                );
                return None;
            }
        };
        let active_memory_bytes = u64::try_from(memory_snapshot.active_memory_bytes()).ok()?;
        let allocator_cache_memory_bytes =
            u64::try_from(memory_snapshot.allocator_cache_memory_bytes()).ok()?;
        let peak_memory_bytes = u64::try_from(memory_snapshot.peak_memory_bytes()).ok()?;
        let expert_payload_bytes = model.weights.expert_payload_bytes();
        let model_core_payload_bytes = model.weights.model_core_payload_bytes();
        let context_state_payload_bytes = self.active.as_ref().map_or(0, |active_generation| {
            context_state_payload_bytes(&active_generation.caches)
        });
        let active_memory_breakdown = MlxActiveMemoryBreakdown::reconcile(
            active_memory_bytes,
            expert_payload_bytes,
            model_core_payload_bytes,
            context_state_payload_bytes,
        );
        let expert_residency_telemetry = ExpertResidencyTelemetry {
            total_layer_count: u32::try_from(model.config.num_hidden_layers()).unwrap_or(u32::MAX),
            resident_expert_count: resident_expert_count(&model.config),
            resident_expert_payload_bytes: active_memory_breakdown.expert_payload_bytes,
        };
        Some(
            MlxMemoryTelemetry::new(
                active_memory_bytes,
                allocator_cache_memory_bytes,
                peak_memory_bytes,
                active_memory_breakdown,
            )
            .with_expert_residency_telemetry(expert_residency_telemetry),
        )
    }

    pub(super) fn minimum_mlx_memory_ceiling_bytes(&self) -> u64 {
        self.model.as_ref().map_or(1, |model| {
            model
                .weights
                .expert_payload_bytes()
                .saturating_add(model.weights.model_core_payload_bytes())
                .max(1)
        })
    }
}

fn context_state_payload_bytes(caches: &[K2HorizonMoVAKvState]) -> u64 {
    caches
        .iter()
        .map(|state| match state {
            K2HorizonMoVAKvState::FullPrecision(state) => state.payload_byte_count(),
            K2HorizonMoVAKvState::Quantized(state) => state.payload_byte_count(),
        })
        .fold(0_u64, u64::saturating_add)
}

fn resident_expert_count(config: &K2HorizonMoVAConfig) -> u32 {
    (0..config.num_hidden_layers())
        .map(
            |decoder_layer_index| match config.layer_kind(decoder_layer_index) {
                K2HorizonMoVALayerKind::Dense => 0_u32,
                K2HorizonMoVALayerKind::SparseFeedForward => {
                    u32::try_from(config.num_experts()).unwrap_or(u32::MAX)
                }
                K2HorizonMoVALayerKind::SparseMixtureOfValues => u32::try_from(
                    config
                        .num_experts()
                        .saturating_add(config.mova_num_experts()),
                )
                .unwrap_or(u32::MAX),
            },
        )
        .fold(0_u32, u32::saturating_add)
}
