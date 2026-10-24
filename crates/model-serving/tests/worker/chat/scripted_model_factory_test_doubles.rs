//! Scripted lazy-model factories shared by engine-backed worker lifecycle tests.

use super::scripted_chat_test_doubles::{ScriptedChatEngine, ScriptedChatProcessor};
use astronomical_ipc_protocol::{ExpertMemoryMode, WorkerModelConfiguration};
use astronomical_model_serving::{
    EnginePlacement, MlxMemoryTelemetry, ModelFactory, ModelFactoryRuntime,
};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

pub(super) struct LazyScriptedModelFactory {
    pub(super) model_factory_call_count: Arc<AtomicUsize>,
    pub(super) mlx_memory_limits: (u64, u64),
    pub(super) model_creation_memory_limits: Arc<Mutex<Vec<(u64, u64)>>>,
    pub(super) model_configurations: Arc<Mutex<Vec<WorkerModelConfiguration>>>,
    /// Lets lifecycle tests prove readiness propagation without model-serving.
    pub(super) expert_memory_mode: Option<ExpertMemoryMode>,
    /// Lets lifecycle tests prove memory-snapshot parity without model-serving.
    pub(super) created_engine_idle_mlx_memory_telemetry: Option<MlxMemoryTelemetry>,
    /// Scripts a live ceiling the created engine cannot adopt in place.
    pub(super) scripted_replacement_required_reason: Option<String>,
}

pub(super) struct FirstCreationFailsScriptedModelFactory {
    pub(super) model_factory_call_count: Arc<AtomicUsize>,
}

impl ModelFactory<ScriptedChatProcessor, ScriptedChatEngine> for LazyScriptedModelFactory {
    async fn create(
        &self,
        _model_directory: &str,
        model_configuration: WorkerModelConfiguration,
    ) -> Result<ModelFactoryRuntime<ScriptedChatProcessor, ScriptedChatEngine>, String> {
        self.model_factory_call_count.fetch_add(1, Ordering::SeqCst);
        self.model_configurations
            .lock()
            .unwrap_or_else(|poisoned_lock| poisoned_lock.into_inner())
            .push(model_configuration);
        let mut scripted_engine = ScriptedChatEngine::new();
        scripted_engine.initial_expert_memory_mode = self.expert_memory_mode;
        if let Some(created_engine_idle_mlx_memory_telemetry) =
            self.created_engine_idle_mlx_memory_telemetry.clone()
        {
            scripted_engine = scripted_engine
                .with_idle_mlx_memory_telemetry(created_engine_idle_mlx_memory_telemetry);
        }
        let (active_memory_limit_bytes, allocator_cache_memory_limit_bytes) =
            self.mlx_memory_limits;
        scripted_engine.maximum_allocator_cache_memory_limit_bytes =
            allocator_cache_memory_limit_bytes;
        if let Some(scripted_replacement_required_reason) =
            &self.scripted_replacement_required_reason
        {
            scripted_engine = scripted_engine
                .with_scripted_replacement_required_reason(scripted_replacement_required_reason);
        }
        self.model_creation_memory_limits
            .lock()
            .unwrap_or_else(|poisoned_lock| poisoned_lock.into_inner())
            .push((
                active_memory_limit_bytes,
                allocator_cache_memory_limit_bytes,
            ));
        Ok(ModelFactoryRuntime::autoregressive(
            ScriptedChatProcessor::new(),
            scripted_engine,
            EnginePlacement::CompleteResidencyFits,
        ))
    }

    fn update_mlx_memory_limits(
        &mut self,
        effective_mlx_memory_ceiling_bytes: u64,
        allocator_cache_memory_limit_bytes: u64,
    ) {
        self.mlx_memory_limits = (
            effective_mlx_memory_ceiling_bytes,
            allocator_cache_memory_limit_bytes,
        );
    }
}

impl ModelFactory<ScriptedChatProcessor, ScriptedChatEngine>
    for FirstCreationFailsScriptedModelFactory
{
    async fn create(
        &self,
        _model_directory: &str,
        _model_configuration: WorkerModelConfiguration,
    ) -> Result<ModelFactoryRuntime<ScriptedChatProcessor, ScriptedChatEngine>, String> {
        let model_factory_call_number =
            self.model_factory_call_count.fetch_add(1, Ordering::SeqCst);
        if model_factory_call_number == 0 {
            return Err("the scripted first model is invalid".to_owned());
        }
        Ok(ModelFactoryRuntime::autoregressive(
            ScriptedChatProcessor::new(),
            ScriptedChatEngine::new(),
            EnginePlacement::CompleteResidencyFits,
        ))
    }
}
