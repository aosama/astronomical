//! Scripted lazy-model factories shared by engine-backed worker lifecycle tests.

use super::scripted_chat_test_doubles::{ScriptedChatEngine, ScriptedChatProcessor};
use super::*;

pub(super) struct LazyScriptedModelFactory {
    pub(super) model_factory_call_count: Arc<AtomicUsize>,
    pub(super) requested_model_ids: Arc<Mutex<Vec<String>>>,
    pub(super) mlx_memory_limits: (u64, u64),
    pub(super) model_creation_memory_limits: Arc<Mutex<Vec<(u64, u64)>>>,
    /// Lets lifecycle tests prove readiness propagation without model-serving.
    pub(super) expert_memory_mode: Option<ExpertMemoryMode>,
}

pub(super) struct FirstCreationFailsScriptedModelFactory {
    pub(super) model_factory_call_count: Arc<AtomicUsize>,
}

impl ModelFactory<ScriptedChatProcessor, ScriptedChatEngine> for LazyScriptedModelFactory {
    async fn create(
        &self,
        model_id: &str,
        _model_directory: &str,
        _max_output_tokens: u32,
    ) -> Result<(ScriptedChatProcessor, ScriptedChatEngine), String> {
        self.model_factory_call_count.fetch_add(1, Ordering::SeqCst);
        self.requested_model_ids
            .lock()
            .unwrap_or_else(|poisoned_lock| poisoned_lock.into_inner())
            .push(model_id.to_owned());
        let mut scripted_engine = ScriptedChatEngine::new();
        scripted_engine.initial_expert_memory_mode = self.expert_memory_mode;
        let (active_memory_limit_bytes, allocator_cache_memory_limit_bytes) =
            self.mlx_memory_limits;
        scripted_engine.maximum_allocator_cache_memory_limit_bytes =
            allocator_cache_memory_limit_bytes;
        self.model_creation_memory_limits
            .lock()
            .unwrap_or_else(|poisoned_lock| poisoned_lock.into_inner())
            .push((
                active_memory_limit_bytes,
                allocator_cache_memory_limit_bytes,
            ));
        Ok((ScriptedChatProcessor::new(), scripted_engine))
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
        _model_id: &str,
        _model_directory: &str,
        _max_output_tokens: u32,
    ) -> Result<(ScriptedChatProcessor, ScriptedChatEngine), String> {
        let model_factory_call_number =
            self.model_factory_call_count.fetch_add(1, Ordering::SeqCst);
        if model_factory_call_number == 0 {
            return Err("the scripted first model is invalid".to_owned());
        }
        Ok((ScriptedChatProcessor::new(), ScriptedChatEngine::new()))
    }
}
