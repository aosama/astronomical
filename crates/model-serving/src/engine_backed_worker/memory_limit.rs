use astronomical_ipc_protocol::{
    ExpertMemoryMode, ImageGenerationFailureReason, MlxMemorySnapshotSource, ProtocolWriter,
    WorkerEvent,
};
use tokio::io::AsyncWrite;

use super::output;
use super::output::worker_expert_residency_snapshot;
use super::support;
use super::support::{ModelFactory, WorkerRuntimeError};
use crate::memory::telemetry::MlxMemoryLimitAdjustment;
use crate::{
    ImageGenerationEngine, InferenceEngine, InferenceEngineError,
    effective_mlx_memory_ceiling_bytes,
};

use super::{EngineBackedWorker, LoadedRuntime};
use crate::model_generation_processor::ModelGenerationProcessor;

/// Outcome of planning an MLX memory-limit adoption when no model runtime is
/// loaded.
///
/// The planner returns the announcement instead of sending it so the owning
/// command keeps every `send_event` call in its original order.
enum FactoryMemoryLimitOutcome {
    /// No model factory remains, so the ceiling cannot be adopted anywhere.
    FactoryUnavailable,
    /// The factory adopted the ceiling; the parent announces the change.
    /// The event is boxed because `WorkerEvent` dwarfs this zero-field variant.
    LimitAdopted(Box<WorkerEvent>),
}

impl<Processor, Engine, Factory, ImageEngine, EmbeddingsEngine>
    EngineBackedWorker<Processor, Engine, Factory, ImageEngine, EmbeddingsEngine>
where
    Processor: ModelGenerationProcessor + Send + 'static,
    Engine: InferenceEngine<Request = Processor::InferenceRequest> + Send + 'static,
    Factory: ModelFactory<Processor, Engine, ImageEngine, EmbeddingsEngine> + Send + 'static,
    ImageEngine: ImageGenerationEngine,
    EmbeddingsEngine: crate::EmbeddingEngine,
{
    pub(crate) async fn update_mlx_memory_limit<WriteTransport>(
        &mut self,
        requested_mlx_memory_ceiling_bytes: u64,
        configuration_generation: String,
        event_writer: &mut ProtocolWriter<WriteTransport>,
    ) -> Result<(), WorkerRuntimeError>
    where
        WriteTransport: AsyncWrite + Unpin,
    {
        if is_mlx_memory_limit_request_out_of_bounds(
            requested_mlx_memory_ceiling_bytes,
            self.machine_mlx_memory_ceiling_bytes,
        ) {
            return self
                .emit_mlx_memory_limit_rejection(
                    requested_mlx_memory_ceiling_bytes,
                    "requested memory ceiling is outside the worker machine limit",
                    event_writer,
                )
                .await;
        }
        // The wired-residency budget is claimed once for the worker lifetime,
        // so a live raise may never plan past it: admission above the wired
        // budget is the issue #1297 gap where Metal evicts GPU-resident
        // weights and the worker dies with an out-of-memory command buffer.
        let effective_requested_mlx_memory_ceiling_bytes =
            effective_requested_ceiling_after_wired_claim(
                requested_mlx_memory_ceiling_bytes,
                self.machine_mlx_memory_ceiling_bytes,
                self.wired_residency_budget_bytes,
            );
        let Some(loaded_runtime) = self.loaded_runtime.as_mut() else {
            return match self.plan_factory_memory_limit_adoption(
                effective_requested_mlx_memory_ceiling_bytes,
                configuration_generation,
            ) {
                FactoryMemoryLimitOutcome::FactoryUnavailable => {
                    self.emit_mlx_memory_limit_rejection(
                        requested_mlx_memory_ceiling_bytes,
                        "model factory is unavailable",
                        event_writer,
                    )
                    .await
                }
                FactoryMemoryLimitOutcome::LimitAdopted(mlx_memory_limit_changed_event) => {
                    event_writer
                        .send_event(&mlx_memory_limit_changed_event)
                        .await?;
                    Ok(())
                }
            };
        };
        let memory_limit_adjustment = Self::adjust_loaded_runtime_mlx_memory_limit(
            loaded_runtime,
            effective_requested_mlx_memory_ceiling_bytes,
        )
        .await;
        match memory_limit_adjustment {
            Ok(mlx_memory_limit_adjustment) => {
                let mlx_memory_limit_changed_event = self.adopt_runtime_memory_limit_adjustment(
                    mlx_memory_limit_adjustment,
                    configuration_generation,
                );
                event_writer
                    .send_event(&mlx_memory_limit_changed_event)
                    .await?;
                Ok(())
            }
            Err(
                InferenceEngineError::ResidentStreamingReplacementRequired { reason }
                | InferenceEngineError::StreamingResidentPromotionRequired { reason },
            ) => {
                self.emit_mlx_memory_limit_restart_required(
                    requested_mlx_memory_ceiling_bytes,
                    effective_requested_mlx_memory_ceiling_bytes,
                    configuration_generation,
                    &reason,
                    event_writer,
                )
                .await
            }
            Err(InferenceEngineError::MlxMemoryLimitRejected {
                minimum_mlx_memory_ceiling_bytes,
                reason,
                ..
            }) => {
                self.minimum_mlx_memory_ceiling_bytes = minimum_mlx_memory_ceiling_bytes;
                self.emit_mlx_memory_limit_rejection(
                    requested_mlx_memory_ceiling_bytes,
                    &reason,
                    event_writer,
                )
                .await
            }
            Err(InferenceEngineError::EngineBusy) => {
                self.emit_mlx_memory_limit_rejection(
                    requested_mlx_memory_ceiling_bytes,
                    "memory limits cannot change during generation",
                    event_writer,
                )
                .await
            }
            Err(engine_error) => Err(support::engine_generation_error(engine_error)),
        }
    }

    fn record_configuration_generation(&mut self, configuration_generation: String) {
        if let Some(worker_runtime_feature_configuration) =
            self.worker_runtime_feature_configuration.as_mut()
        {
            worker_runtime_feature_configuration.configuration_generation =
                configuration_generation;
        }
    }

    /// Dispatches the live memory-limit change to whichever runtime is loaded.
    async fn adjust_loaded_runtime_mlx_memory_limit(
        loaded_runtime: &mut LoadedRuntime<Processor, Engine, ImageEngine, EmbeddingsEngine>,
        effective_requested_mlx_memory_ceiling_bytes: u64,
    ) -> Result<MlxMemoryLimitAdjustment, InferenceEngineError> {
        match loaded_runtime {
            LoadedRuntime::Autoregressive(loaded_model) => {
                loaded_model
                    .engine
                    .update_mlx_memory_limit(effective_requested_mlx_memory_ceiling_bytes)
                    .await
            }
            LoadedRuntime::Image(image_engine) => image_engine
                .update_mlx_memory_limit(effective_requested_mlx_memory_ceiling_bytes)
                .map_err(map_image_memory_limit_failure_to_engine_error),
            LoadedRuntime::Embeddings(_embedding_engine) => Err(InferenceEngineError::Fatal {
                reason: "this embedding engine does not support live MLX memory limits".to_owned(),
            }),
        }
    }

    /// Adopts the requested ceiling through the model factory alone, for a
    /// worker whose runtime has not loaded (or has already been replaced).
    ///
    /// The factory keeps generating models under the new ceiling while the
    /// daemon restarts the worker, so the adoption must land before the
    /// announcement event is built.
    fn plan_factory_memory_limit_adoption(
        &mut self,
        effective_requested_mlx_memory_ceiling_bytes: u64,
        configuration_generation: String,
    ) -> FactoryMemoryLimitOutcome {
        let Some(model_factory) = self.model_factory.as_mut() else {
            return FactoryMemoryLimitOutcome::FactoryUnavailable;
        };
        model_factory.update_mlx_memory_limits(
            effective_requested_mlx_memory_ceiling_bytes,
            effective_requested_mlx_memory_ceiling_bytes,
        );
        self.effective_mlx_memory_ceiling_bytes = effective_requested_mlx_memory_ceiling_bytes;
        self.record_configuration_generation(configuration_generation);
        FactoryMemoryLimitOutcome::LimitAdopted(Box::new(WorkerEvent::MlxMemoryLimitChanged {
            effective_mlx_memory_ceiling_bytes: effective_requested_mlx_memory_ceiling_bytes,
            minimum_mlx_memory_ceiling_bytes: self.minimum_mlx_memory_ceiling_bytes,
            expert_memory_mode: ExpertMemoryMode::Resident,
            mlx_memory_snapshot: None,
            expert_residency: None,
        }))
    }

    /// Applies one successful runtime adjustment to the worker's ceilings,
    /// factory, and configuration generation, and builds the announcement.
    fn adopt_runtime_memory_limit_adjustment(
        &mut self,
        mlx_memory_limit_adjustment: MlxMemoryLimitAdjustment,
        configuration_generation: String,
    ) -> WorkerEvent {
        self.effective_mlx_memory_ceiling_bytes =
            mlx_memory_limit_adjustment.effective_mlx_memory_ceiling_bytes();
        self.minimum_mlx_memory_ceiling_bytes =
            mlx_memory_limit_adjustment.minimum_mlx_memory_ceiling_bytes();
        if let Some(model_factory) = self.model_factory.as_mut() {
            model_factory.update_mlx_memory_limits(
                self.effective_mlx_memory_ceiling_bytes,
                mlx_memory_limit_adjustment.allocator_cache_memory_limit_bytes(),
            );
        }
        self.record_configuration_generation(configuration_generation);
        WorkerEvent::MlxMemoryLimitChanged {
            effective_mlx_memory_ceiling_bytes: self.effective_mlx_memory_ceiling_bytes,
            minimum_mlx_memory_ceiling_bytes: self.minimum_mlx_memory_ceiling_bytes,
            expert_memory_mode: mlx_memory_limit_adjustment.expert_memory_mode(),
            mlx_memory_snapshot: mlx_memory_limit_adjustment.mlx_memory_telemetry().map(
                |mlx_memory_telemetry| {
                    output::worker_memory_snapshot(
                        MlxMemorySnapshotSource::MemoryLimitAdjusted,
                        mlx_memory_telemetry,
                    )
                },
            ),
            expert_residency: mlx_memory_limit_adjustment
                .expert_residency_telemetry()
                .map(worker_expert_residency_snapshot),
        }
    }

    /// Reports that this process cannot adopt the requested ceiling and the
    /// daemon must restart the worker under the persisted lower limit.
    ///
    /// Constructing the streaming replacement here would briefly require the
    /// resident and streaming model runtimes in one process, so MLX memory
    /// could not be released deterministically. The loaded runtime, ceilings,
    /// and configuration generation stay untouched so the worker remains an
    /// honestly consistent server if the daemon never performs the restart.
    async fn emit_mlx_memory_limit_restart_required<WriteTransport>(
        &self,
        requested_mlx_memory_ceiling_bytes: u64,
        effective_mlx_memory_ceiling_bytes: u64,
        configuration_generation: String,
        replacement_reason: &str,
        event_writer: &mut ProtocolWriter<WriteTransport>,
    ) -> Result<(), WorkerRuntimeError>
    where
        WriteTransport: AsyncWrite + Unpin,
    {
        tracing::info!(
            requested_mlx_memory_ceiling_bytes = effective_mlx_memory_ceiling_bytes,
            reason = %replacement_reason,
            "resident engine cannot adopt the requested MLX ceiling in place; daemon restart required"
        );
        event_writer
            .send_event(&WorkerEvent::MlxMemoryLimitRestartRequired {
                requested_mlx_memory_ceiling_bytes,
                effective_mlx_memory_ceiling_bytes,
                minimum_mlx_memory_ceiling_bytes: self.minimum_mlx_memory_ceiling_bytes,
                configuration_generation,
                reason: replacement_reason.chars().take(256).collect(),
            })
            .await?;
        Ok(())
    }

    async fn emit_mlx_memory_limit_rejection<WriteTransport>(
        &self,
        requested_mlx_memory_ceiling_bytes: u64,
        reason: &str,
        event_writer: &mut ProtocolWriter<WriteTransport>,
    ) -> Result<(), WorkerRuntimeError>
    where
        WriteTransport: AsyncWrite + Unpin,
    {
        event_writer
            .send_event(&WorkerEvent::MlxMemoryLimitRejected {
                requested_mlx_memory_ceiling_bytes,
                minimum_mlx_memory_ceiling_bytes: self.minimum_mlx_memory_ceiling_bytes,
                machine_mlx_memory_ceiling_bytes: self.machine_mlx_memory_ceiling_bytes,
                reason: reason.chars().take(256).collect(),
            })
            .await?;
        Ok(())
    }
}

/// Returns whether the requested ceiling must be rejected before planning.
///
/// A zero ceiling cannot host any model, and a ceiling above the machine
/// bound cannot be honored by the Metal device, so both are rejected up
/// front instead of producing a silently unusable worker.
fn is_mlx_memory_limit_request_out_of_bounds(
    requested_mlx_memory_ceiling_bytes: u64,
    machine_mlx_memory_ceiling_bytes: u64,
) -> bool {
    requested_mlx_memory_ceiling_bytes == 0
        || requested_mlx_memory_ceiling_bytes > machine_mlx_memory_ceiling_bytes
}

/// Resolves the ceiling the engine may plan against for this request.
///
/// The request is first capped by the machine bound and then by the
/// plannable wired residency: both bounds keep the residency planner's
/// activation headroom real instead of letting admission evict wired
/// GPU-resident weights (issue #1297).
fn effective_requested_ceiling_after_wired_claim(
    requested_mlx_memory_ceiling_bytes: u64,
    machine_mlx_memory_ceiling_bytes: u64,
    wired_residency_budget_bytes: Option<u64>,
) -> u64 {
    crate::effective_mlx_memory_ceiling_after_wired_claim(
        effective_mlx_memory_ceiling_bytes(
            Some(requested_mlx_memory_ceiling_bytes),
            machine_mlx_memory_ceiling_bytes,
        ),
        wired_residency_budget_bytes,
    )
}

/// Maps an image engine's memory-limit failure onto the shared engine error.
///
/// Only `EngineBusy` is recoverable protocol state; every other image
/// failure is surfaced as fatal because the image engine cannot promise a
/// consistent limit change afterwards.
fn map_image_memory_limit_failure_to_engine_error(
    failure_reason: ImageGenerationFailureReason,
) -> InferenceEngineError {
    match failure_reason {
        ImageGenerationFailureReason::EngineBusy => InferenceEngineError::EngineBusy,
        ImageGenerationFailureReason::InvalidRequest { reason }
        | ImageGenerationFailureReason::EncodingFailed { reason }
        | ImageGenerationFailureReason::FatalExecution { reason } => {
            InferenceEngineError::Fatal { reason }
        }
        other => InferenceEngineError::Fatal {
            reason: format!("image memory-limit update failed: {other:?}"),
        },
    }
}

#[cfg(test)]
#[path = "memory_limit_tests.rs"]
mod memory_limit_tests;
