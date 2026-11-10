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

impl<Processor, Engine, Factory, ImageEngine, EmbeddingsEngine, DecisionsEngine>
    EngineBackedWorker<Processor, Engine, Factory, ImageEngine, EmbeddingsEngine, DecisionsEngine>
where
    Processor: ModelGenerationProcessor + Send + 'static,
    Engine: InferenceEngine<Request = Processor::InferenceRequest> + Send + 'static,
    Factory: ModelFactory<Processor, Engine, ImageEngine, EmbeddingsEngine, DecisionsEngine>
        + Send
        + 'static,
    ImageEngine: ImageGenerationEngine,
    EmbeddingsEngine: crate::EmbeddingEngine,
    DecisionsEngine: crate::DecisionEngine,
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
        let live_limit_plan =
            self.plan_live_memory_limit_update(requested_mlx_memory_ceiling_bytes);
        let effective_requested_mlx_memory_ceiling_bytes = live_limit_plan.effective_ceiling_bytes;
        if let Some(rejection_reason) = self.mlx_memory_limit_rejection_before_planning(
            requested_mlx_memory_ceiling_bytes,
            effective_requested_mlx_memory_ceiling_bytes,
        ) {
            return self
                .emit_mlx_memory_limit_rejection(
                    requested_mlx_memory_ceiling_bytes,
                    rejection_reason,
                    event_writer,
                )
                .await;
        }
        let Some(loaded_runtime) = self.loaded_runtime.as_mut() else {
            return match self
                .plan_factory_memory_limit_adoption(live_limit_plan, configuration_generation)
            {
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
                    live_limit_plan,
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

    /// Serves the just-loaded model at its floor when its floor is above the
    /// ceiling and raising it would not pass a limit the user set (#1902),
    /// and returns the announcement of the raised ceiling.
    ///
    /// Engines report their floor at load and size every request against the
    /// ceiling they were given, so without this a dense model larger than an
    /// 8 GB Mac's wired budget loads and then refuses even a one-token request.
    /// The new engine was built at the factory's ceiling, so that is where the
    /// worker's own ceiling starts, not at a floor an earlier model was raised
    /// to. A raise the engine cannot adopt is logged and the model keeps the
    /// factory's ceiling.
    pub(super) async fn serve_loaded_model_at_its_floor_when_allowed(
        &mut self,
    ) -> Option<WorkerEvent> {
        self.effective_mlx_memory_ceiling_bytes = self.factory_mlx_memory_ceiling_bytes;
        let floor_ceiling_bytes = crate::floor_raised_ceiling_bytes(
            self.effective_mlx_memory_ceiling_bytes,
            self.minimum_mlx_memory_ceiling_bytes,
            self.floor_raise_limit_bytes,
        )?;
        let loaded_runtime = self.loaded_runtime.as_mut()?;
        match Self::adjust_loaded_runtime_mlx_memory_limit(loaded_runtime, floor_ceiling_bytes)
            .await
        {
            Ok(mlx_memory_limit_adjustment) => {
                tracing::info!(
                    floor_ceiling_bytes,
                    "serving the loaded model at its floor above the ceiling"
                );
                Some(self.memory_limit_changed_event(&mlx_memory_limit_adjustment))
            }
            Err(raise_error) => {
                tracing::warn!(
                    error = %raise_error,
                    floor_ceiling_bytes,
                    "the loaded model could not be raised to its floor; it keeps its ceiling"
                );
                None
            }
        }
    }

    /// Resolves a live update before anything changes: the requested ceiling
    /// after Metal's wired claim (the wired budget is claimed once for the
    /// worker lifetime, and admission above it is the #1297 gap where Metal
    /// evicts GPU-resident weights), the user's raise limit derived from the
    /// request, and the ceiling the loaded model would then run at.
    fn plan_live_memory_limit_update(
        &self,
        requested_mlx_memory_ceiling_bytes: u64,
    ) -> LiveMemoryLimitPlan {
        let floor_raise_limit_bytes = crate::floor_raise_limit_bytes(
            Some(requested_mlx_memory_ceiling_bytes),
            self.machine_mlx_memory_ceiling_bytes,
        );
        let wired_clamped_ceiling_bytes = effective_requested_ceiling_after_wired_claim(
            requested_mlx_memory_ceiling_bytes,
            self.machine_mlx_memory_ceiling_bytes,
            self.wired_residency_budget_bytes,
        );
        LiveMemoryLimitPlan {
            wired_clamped_ceiling_bytes,
            effective_ceiling_bytes: self
                .live_update_ceiling_bytes(wired_clamped_ceiling_bytes, floor_raise_limit_bytes),
            floor_raise_limit_bytes,
        }
    }

    /// The ceiling a live update applies: the requested ceiling after Metal's
    /// wired clamp, raised to the loaded model's floor when that stays within
    /// `floor_raise_limit_bytes`. Without the raise, choosing "Use Max" on a
    /// model served at its floor would push it back under that floor.
    fn live_update_ceiling_bytes(
        &self,
        wired_clamped_ceiling_bytes: u64,
        floor_raise_limit_bytes: u64,
    ) -> u64 {
        if self.loaded_runtime.is_none() {
            return wired_clamped_ceiling_bytes;
        }
        crate::floor_raised_ceiling_bytes(
            wired_clamped_ceiling_bytes,
            self.minimum_mlx_memory_ceiling_bytes,
            floor_raise_limit_bytes,
        )
        .unwrap_or(wired_clamped_ceiling_bytes)
    }

    /// Keeps the raise limit of an adopted live update, and hands it to the
    /// factory so the next model's load plans by the same rule. Only the two
    /// adoption paths call this, so a rejected update leaves the previous
    /// limit in place.
    fn adopt_floor_raise_limit(&mut self, floor_raise_limit_bytes: u64) {
        self.floor_raise_limit_bytes = floor_raise_limit_bytes;
        if let Some(model_factory) = self.model_factory.as_mut() {
            model_factory.update_floor_raise_limit(floor_raise_limit_bytes);
        }
    }

    /// Returns why a requested ceiling must be refused before any runtime sees it.
    ///
    /// Image engines plan their whole execution against the ceiling, so one
    /// below the floor they reported at load cannot run at all, and their own
    /// planners report that as a fatal failure that ends the worker (#1899).
    /// Refusing it here instead keeps the loaded image model serving, the same
    /// way chat engines already refuse a ceiling below their streaming floor.
    fn mlx_memory_limit_rejection_before_planning(
        &self,
        requested_mlx_memory_ceiling_bytes: u64,
        effective_requested_mlx_memory_ceiling_bytes: u64,
    ) -> Option<&'static str> {
        if is_mlx_memory_limit_request_out_of_bounds(
            requested_mlx_memory_ceiling_bytes,
            self.machine_mlx_memory_ceiling_bytes,
        ) {
            return Some("requested memory ceiling is outside the worker machine limit");
        }
        let has_loaded_image_runtime = matches!(self.loaded_runtime, Some(LoadedRuntime::Image(_)));
        (has_loaded_image_runtime
            && effective_requested_mlx_memory_ceiling_bytes < self.minimum_mlx_memory_ceiling_bytes)
            .then_some("the requested ceiling is below the image model's execution floor")
    }

    /// Dispatches the live memory-limit change to whichever runtime is loaded.
    async fn adjust_loaded_runtime_mlx_memory_limit(
        loaded_runtime: &mut LoadedRuntime<
            Processor,
            Engine,
            ImageEngine,
            EmbeddingsEngine,
            DecisionsEngine,
        >,
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
            // Both answer in one forward pass and hold no state between
            // requests, so neither has a live limit to move.
            LoadedRuntime::Embeddings(_) | LoadedRuntime::Decisions(_) => {
                Err(InferenceEngineError::Fatal {
                    reason: "embedding and decision engines do not support live MLX memory limits"
                        .to_owned(),
                })
            }
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
        live_limit_plan: LiveMemoryLimitPlan,
        configuration_generation: String,
    ) -> FactoryMemoryLimitOutcome {
        let effective_requested_mlx_memory_ceiling_bytes = live_limit_plan.effective_ceiling_bytes;
        let Some(model_factory) = self.model_factory.as_mut() else {
            return FactoryMemoryLimitOutcome::FactoryUnavailable;
        };
        model_factory.update_mlx_memory_limits(
            effective_requested_mlx_memory_ceiling_bytes,
            effective_requested_mlx_memory_ceiling_bytes,
        );
        self.effective_mlx_memory_ceiling_bytes = effective_requested_mlx_memory_ceiling_bytes;
        self.factory_mlx_memory_ceiling_bytes = live_limit_plan.wired_clamped_ceiling_bytes;
        self.adopt_floor_raise_limit(live_limit_plan.floor_raise_limit_bytes);
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
    ///
    /// The factory receives the wired-clamped ceiling, never the floor a
    /// loaded model was raised to, so the next model is planned against the
    /// machine and not above Metal's wired claim (#1297).
    fn adopt_runtime_memory_limit_adjustment(
        &mut self,
        mlx_memory_limit_adjustment: MlxMemoryLimitAdjustment,
        configuration_generation: String,
        live_limit_plan: LiveMemoryLimitPlan,
    ) -> WorkerEvent {
        let factory_ceiling_bytes = live_limit_plan.wired_clamped_ceiling_bytes;
        if let Some(model_factory) = self.model_factory.as_mut() {
            model_factory.update_mlx_memory_limits(
                factory_ceiling_bytes,
                mlx_memory_limit_adjustment
                    .allocator_cache_memory_limit_bytes()
                    .min(factory_ceiling_bytes),
            );
        }
        self.factory_mlx_memory_ceiling_bytes = factory_ceiling_bytes;
        self.adopt_floor_raise_limit(live_limit_plan.floor_raise_limit_bytes);
        self.record_configuration_generation(configuration_generation);
        self.memory_limit_changed_event(&mlx_memory_limit_adjustment)
    }

    /// Records one adopted adjustment's ceiling and floor on the worker and
    /// builds the announcement the daemon publishes to status.
    fn memory_limit_changed_event(
        &mut self,
        mlx_memory_limit_adjustment: &MlxMemoryLimitAdjustment,
    ) -> WorkerEvent {
        self.effective_mlx_memory_ceiling_bytes =
            mlx_memory_limit_adjustment.effective_mlx_memory_ceiling_bytes();
        self.minimum_mlx_memory_ceiling_bytes =
            mlx_memory_limit_adjustment.minimum_mlx_memory_ceiling_bytes();
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

/// One live memory-limit update resolved before any state changes.
#[derive(Clone, Copy, Debug)]
pub(super) struct LiveMemoryLimitPlan {
    /// The requested ceiling after the machine bound and Metal's wired claim.
    wired_clamped_ceiling_bytes: u64,
    /// The ceiling the loaded model would run at: the wired-clamped ceiling,
    /// or the model's floor when the user set no limit below it.
    effective_ceiling_bytes: u64,
    /// The user's limit derived from this request (see `floor_raise_limit_bytes`).
    floor_raise_limit_bytes: u64,
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
