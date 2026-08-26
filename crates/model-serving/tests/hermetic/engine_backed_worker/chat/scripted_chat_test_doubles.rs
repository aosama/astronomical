use super::*;

pub(crate) struct ScriptedChatProcessor {
    emits_parallel_tool_calls: bool,
    prompt_token_count: usize,
}

pub(super) struct MalformedFinishProcessor;
pub(super) struct CorrectionRequestingProcessor;

impl ScriptedChatProcessor {
    pub(crate) fn new() -> Self {
        Self {
            emits_parallel_tool_calls: false,
            prompt_token_count: 1,
        }
    }

    pub(super) fn with_parallel_tool_calls() -> Self {
        Self {
            emits_parallel_tool_calls: true,
            prompt_token_count: 1,
        }
    }

    pub(super) fn with_prompt_token_count(prompt_token_count: usize) -> Self {
        Self {
            emits_parallel_tool_calls: false,
            prompt_token_count,
        }
    }
}

impl ModelGenerationProcessor for MalformedFinishProcessor {
    type InferenceRequest = ScriptedInferenceRequest;
    type RequestOutput = ();

    fn ready_event(
        &self,
        mtp_runtime_state: MtpRuntimeState,
        mtp_unavailable_reason: Option<String>,
        _mtp_depth_status: astronomical_ipc_protocol::MtpDepthStatus,
        speculative_prefill_runtime_state: SpeculativePrefillRuntimeState,
        speculative_prefill_unavailable_reason: Option<String>,
        speculative_prefill_draft_model_id: Option<String>,
        speculative_prefill_draft_model_revision: Option<String>,
    ) -> WorkerEvent {
        ready_event_with_speculative_prefill_load_details(
            mtp_runtime_state,
            mtp_unavailable_reason,
            speculative_prefill_runtime_state,
            speculative_prefill_unavailable_reason,
            speculative_prefill_draft_model_id,
            speculative_prefill_draft_model_revision,
        )
    }

    fn prepare_chat_generation(
        &self,
        generation_command: &ChatGenerationCommand,
    ) -> Result<
        PreparedModelGeneration<Self::InferenceRequest, Self::RequestOutput>,
        ChatGenerationFailureReason,
    > {
        Ok(prepared_generation(generation_command, 1))
    }

    fn is_end_of_sequence_token(&self, generated_token_id: u32) -> bool {
        generated_token_id == 1
    }

    fn translate_generated_token(
        &self,
        _request_output: &mut Self::RequestOutput,
        _generated_token_id: u32,
    ) -> Result<ModelGeneratedTokenTranslation, ModelGenerationOutputError> {
        Ok(ModelGeneratedTokenTranslation::from_outputs(Vec::new()))
    }

    fn finish_request_output(
        &self,
        _request_output: &mut Self::RequestOutput,
    ) -> Result<Vec<ChatGenerationOutput>, ModelGenerationOutputError> {
        Err(ModelGenerationOutputError::MalformedOutput {
            diagnostic: Box::new(astronomical_model_serving::MalformedModelOutputDiagnostic {
                diagnostic_code: "scripted_malformed_finish",
                parser_error: "scripted malformed finish".to_owned(),
                generated_token_ids: Vec::new(),
                pending_token_ids: Vec::new(),
                decoded_output_text: String::new(),
                parser_state: "scripted",
                parser_pending_output_text: String::new(),
            }),
        })
    }
}

impl ModelGenerationProcessor for ScriptedChatProcessor {
    type InferenceRequest = ScriptedInferenceRequest;
    type RequestOutput = ();

    fn ready_event(
        &self,
        mtp_runtime_state: MtpRuntimeState,
        mtp_unavailable_reason: Option<String>,
        _mtp_depth_status: astronomical_ipc_protocol::MtpDepthStatus,
        speculative_prefill_runtime_state: SpeculativePrefillRuntimeState,
        speculative_prefill_unavailable_reason: Option<String>,
        speculative_prefill_draft_model_id: Option<String>,
        speculative_prefill_draft_model_revision: Option<String>,
    ) -> WorkerEvent {
        ready_event_with_speculative_prefill_load_details(
            mtp_runtime_state,
            mtp_unavailable_reason,
            speculative_prefill_runtime_state,
            speculative_prefill_unavailable_reason,
            speculative_prefill_draft_model_id,
            speculative_prefill_draft_model_revision,
        )
    }

    fn prepare_chat_generation(
        &self,
        generation_command: &ChatGenerationCommand,
    ) -> Result<
        PreparedModelGeneration<Self::InferenceRequest, Self::RequestOutput>,
        ChatGenerationFailureReason,
    > {
        Ok(prepared_generation(
            generation_command,
            self.prompt_token_count,
        ))
    }

    fn is_end_of_sequence_token(&self, _generated_token_id: u32) -> bool {
        false
    }

    fn translate_generated_token(
        &self,
        _request_output: &mut Self::RequestOutput,
        generated_token_id: u32,
    ) -> Result<ModelGeneratedTokenTranslation, ModelGenerationOutputError> {
        match generated_token_id {
            1 => Ok(ModelGeneratedTokenTranslation::from_outputs(vec![
                ChatGenerationOutput::Reasoning {
                    text: "I should inspect the source tree.".to_owned(),
                },
                ChatGenerationOutput::Text {
                    text: "I found Rust files.".to_owned(),
                },
            ])),
            2 => Ok(ModelGeneratedTokenTranslation::from_outputs(vec![
                ChatGenerationOutput::ToolCall {
                    tool_call_index: 0,
                    function_name: "glob".to_owned(),
                    arguments_json: r#"{"pattern":"src/**/*.rs"}"#.to_owned(),
                },
            ])),
            3 if self.emits_parallel_tool_calls => {
                Ok(ModelGeneratedTokenTranslation::from_outputs(vec![
                    ChatGenerationOutput::ToolCall {
                        tool_call_index: 1,
                        function_name: "glob".to_owned(),
                        arguments_json: r#"{"pattern":"tests/**/*.rs"}"#.to_owned(),
                    },
                ]))
            }
            _ => Err(ModelGenerationOutputError::Fatal {
                reason: "unexpected scripted token".to_owned(),
            }),
        }
    }

    fn finish_request_output(
        &self,
        _request_output: &mut Self::RequestOutput,
    ) -> Result<Vec<ChatGenerationOutput>, ModelGenerationOutputError> {
        Ok(Vec::new())
    }
}

impl ModelGenerationProcessor for CorrectionRequestingProcessor {
    type InferenceRequest = ScriptedInferenceRequest;
    type RequestOutput = ();

    fn ready_event(
        &self,
        mtp_runtime_state: MtpRuntimeState,
        mtp_unavailable_reason: Option<String>,
        _mtp_depth_status: astronomical_ipc_protocol::MtpDepthStatus,
        speculative_prefill_runtime_state: SpeculativePrefillRuntimeState,
        speculative_prefill_unavailable_reason: Option<String>,
        speculative_prefill_draft_model_id: Option<String>,
        speculative_prefill_draft_model_revision: Option<String>,
    ) -> WorkerEvent {
        ready_event_with_speculative_prefill_load_details(
            mtp_runtime_state,
            mtp_unavailable_reason,
            speculative_prefill_runtime_state,
            speculative_prefill_unavailable_reason,
            speculative_prefill_draft_model_id,
            speculative_prefill_draft_model_revision,
        )
    }

    fn prepare_chat_generation(
        &self,
        generation_command: &ChatGenerationCommand,
    ) -> Result<
        PreparedModelGeneration<Self::InferenceRequest, Self::RequestOutput>,
        ChatGenerationFailureReason,
    > {
        Ok(prepared_generation(generation_command, 1))
    }

    fn is_end_of_sequence_token(&self, _generated_token_id: u32) -> bool {
        false
    }

    fn translate_generated_token(
        &self,
        _request_output: &mut Self::RequestOutput,
        generated_token_id: u32,
    ) -> Result<ModelGeneratedTokenTranslation, ModelGenerationOutputError> {
        match generated_token_id {
            1 => Ok(ModelGeneratedTokenTranslation::new(
                vec![ChatGenerationOutput::Reasoning {
                    text: "I should use a tool.".to_owned(),
                }],
                vec![81, 82, 83],
            )),
            2 => Ok(ModelGeneratedTokenTranslation::from_outputs(vec![
                ChatGenerationOutput::ToolCall {
                    tool_call_index: 0,
                    function_name: "glob".to_owned(),
                    arguments_json: r#"{"pattern":"src/**/*.rs"}"#.to_owned(),
                },
            ])),
            _ => Err(ModelGenerationOutputError::Fatal {
                reason: "unexpected scripted token".to_owned(),
            }),
        }
    }

    fn finish_request_output(
        &self,
        _request_output: &mut Self::RequestOutput,
    ) -> Result<Vec<ChatGenerationOutput>, ModelGenerationOutputError> {
        Ok(Vec::new())
    }
}

fn prepared_generation(
    _generation_command: &ChatGenerationCommand,
    prompt_token_count: usize,
) -> PreparedModelGeneration<ScriptedInferenceRequest, ()> {
    PreparedModelGeneration::new(ScriptedInferenceRequest::new(prompt_token_count), ())
}

pub(crate) struct ScriptedChatEngine {
    cancellation_count: Arc<AtomicUsize>,
    cached_token_count: u32,
    restored_prompt_prefix_token_count: u32,
    generated_tokens: Vec<GeneratedToken>,
    is_active: bool,
    next_token_index: usize,
    fatal_decode_reason: Option<String>,
    pub(super) initial_expert_memory_mode: Option<ExpertMemoryMode>,
    cancelled_generation_finalization: GenerationFinalization,
    mtp_runtime_state: MtpRuntimeState,
    mtp_unavailable_reason: Option<String>,
    speculative_prefill_runtime_state: SpeculativePrefillRuntimeState,
    speculative_prefill_unavailable_reason: Option<String>,
    speculative_prefill_draft_model_id: Option<String>,
    speculative_prefill_draft_model_revision: Option<String>,
    active_generation_prompt_cache_stats: Option<WorkerEvent>,
    prompt_cache_clear_event: Option<WorkerEvent>,
    pub(super) maximum_allocator_cache_memory_limit_bytes: u64,
}

impl ScriptedChatEngine {
    pub(crate) fn new() -> Self {
        Self::with_cached_token_count(0)
    }

    pub(super) fn with_cached_token_count(cached_token_count: u32) -> Self {
        Self::with_cached_token_count_and_generated_tokens(
            cached_token_count,
            vec![
                GeneratedToken::TokenId {
                    token_id: 1,
                    is_reasoning_token: false,
                    expert_memory_mode: None,
                    mlx_memory_telemetry: None,
                    first_decode_forward_elapsed_millis: None,
                    generation_finalization: None,
                },
                GeneratedToken::TokenId {
                    token_id: 2,
                    is_reasoning_token: false,
                    expert_memory_mode: None,
                    mlx_memory_telemetry: None,
                    first_decode_forward_elapsed_millis: None,
                    generation_finalization: None,
                },
            ],
        )
    }

    pub(super) fn with_cached_token_count_and_generated_tokens(
        cached_token_count: u32,
        generated_tokens: Vec<GeneratedToken>,
    ) -> Self {
        Self {
            cancellation_count: Arc::new(AtomicUsize::new(0)),
            cached_token_count,
            restored_prompt_prefix_token_count: cached_token_count,
            generated_tokens,
            is_active: false,
            next_token_index: 0,
            fatal_decode_reason: None,
            initial_expert_memory_mode: None,
            cancelled_generation_finalization: GenerationFinalization::default(),
            mtp_runtime_state: MtpRuntimeState::Disabled,
            mtp_unavailable_reason: None,
            speculative_prefill_runtime_state: SpeculativePrefillRuntimeState::Disabled,
            speculative_prefill_unavailable_reason: None,
            speculative_prefill_draft_model_id: None,
            speculative_prefill_draft_model_revision: None,
            active_generation_prompt_cache_stats: None,
            prompt_cache_clear_event: None,
            maximum_allocator_cache_memory_limit_bytes: u64::MAX,
        }
    }

    pub(super) fn with_restored_prompt_prefix_token_count(
        mut self,
        restored_prompt_prefix_token_count: u32,
    ) -> Self {
        self.restored_prompt_prefix_token_count = restored_prompt_prefix_token_count;
        self
    }

    pub(super) fn with_fatal_decode_reason(fatal_decode_reason: &str) -> Self {
        let mut scripted_chat_engine =
            Self::with_cached_token_count_and_generated_tokens(0, Vec::new());
        scripted_chat_engine.fatal_decode_reason = Some(fatal_decode_reason.to_owned());
        scripted_chat_engine
    }

    pub(super) fn with_cancelled_generation_finalization(
        expert_memory_mode: ExpertMemoryMode,
        active_memory_bytes: u64,
        allocator_cache_memory_bytes: u64,
        peak_memory_bytes: u64,
        expert_payload_bytes: u64,
        model_core_payload_bytes: u64,
        context_state_payload_bytes: u64,
    ) -> Self {
        let mut scripted_engine = Self::new();
        scripted_engine.cancelled_generation_finalization = GenerationFinalization::new(
            Some(expert_memory_mode),
            Some(MlxMemoryTelemetry::new(
                active_memory_bytes,
                allocator_cache_memory_bytes,
                peak_memory_bytes,
                MlxActiveMemoryBreakdown {
                    expert_payload_bytes,
                    model_core_payload_bytes,
                    context_state_payload_bytes,
                    speculative_prefill_draft_memory_bytes: 0,
                },
            )),
            None,
        );
        scripted_engine
    }

    pub(super) fn with_mtp_runtime_state(
        mut self,
        mtp_runtime_state: MtpRuntimeState,
        mtp_unavailable_reason: Option<String>,
    ) -> Self {
        self.mtp_runtime_state = mtp_runtime_state;
        self.mtp_unavailable_reason = mtp_unavailable_reason;
        self
    }

    pub(super) fn with_speculative_prefill_runtime(
        mut self,
        speculative_prefill_runtime_state: SpeculativePrefillRuntimeState,
        speculative_prefill_unavailable_reason: Option<String>,
        speculative_prefill_draft_model_id: Option<String>,
        speculative_prefill_draft_model_revision: Option<String>,
    ) -> Self {
        self.speculative_prefill_runtime_state = speculative_prefill_runtime_state;
        self.speculative_prefill_unavailable_reason = speculative_prefill_unavailable_reason;
        self.speculative_prefill_draft_model_id = speculative_prefill_draft_model_id;
        self.speculative_prefill_draft_model_revision = speculative_prefill_draft_model_revision;
        self
    }

    pub(super) fn cancellation_count(&self) -> Arc<AtomicUsize> {
        Arc::clone(&self.cancellation_count)
    }

    pub(super) fn with_active_generation_prompt_cache_stats(
        mut self,
        active_generation_prompt_cache_stats: WorkerEvent,
    ) -> Self {
        self.active_generation_prompt_cache_stats = Some(active_generation_prompt_cache_stats);
        self
    }

    pub(super) fn with_prompt_cache_clear_event(
        mut self,
        prompt_cache_clear_event: WorkerEvent,
    ) -> Self {
        self.prompt_cache_clear_event = Some(prompt_cache_clear_event);
        self
    }
}

impl InferenceEngine for ScriptedChatEngine {
    type Request = ScriptedInferenceRequest;

    async fn load(&mut self) -> Result<EngineLoadResult, InferenceEngineError> {
        let mut engine_load_result = EngineLoadResult::new()
            .with_expert_memory_mode(self.initial_expert_memory_mode)
            .with_mtp_runtime_state(self.mtp_runtime_state)
            .with_speculative_prefill_runtime(
                self.speculative_prefill_runtime_state,
                self.speculative_prefill_unavailable_reason.clone(),
                self.speculative_prefill_draft_model_id.clone(),
                self.speculative_prefill_draft_model_revision.clone(),
            );
        if let Some(mtp_unavailable_reason) = self.mtp_unavailable_reason.clone() {
            engine_load_result =
                engine_load_result.with_mtp_unavailable_reason(mtp_unavailable_reason);
        }
        Ok(engine_load_result)
    }

    async fn start_generation(
        &mut self,
        _generation_request: ScriptedInferenceRequest,
    ) -> Result<EngineGenerationStart, InferenceEngineError> {
        if self.is_active {
            return Err(InferenceEngineError::EngineBusy);
        }
        self.is_active = true;
        self.next_token_index = 0;
        Ok(self
            .initial_expert_memory_mode
            .map_or_else(
                || EngineGenerationStart::new(self.cached_token_count),
                |expert_memory_mode| {
                    EngineGenerationStart::with_expert_memory_mode(
                        self.cached_token_count,
                        expert_memory_mode,
                    )
                },
            )
            .with_restored_prompt_prefix_token_count(self.restored_prompt_prefix_token_count))
    }

    async fn decode_next_token(
        &mut self,
        _request_id: RequestId,
    ) -> Result<GeneratedToken, InferenceEngineError> {
        if let Some(fatal_decode_reason) = &self.fatal_decode_reason {
            return Err(InferenceEngineError::Fatal {
                reason: fatal_decode_reason.clone(),
            });
        }
        let generated_token = self
            .generated_tokens
            .get(self.next_token_index)
            .cloned()
            .unwrap_or(GeneratedToken::EndOfSequence);
        self.next_token_index += 1;
        if matches!(
            generated_token,
            GeneratedToken::TokenId {
                generation_finalization: Some(_),
                ..
            }
        ) {
            self.is_active = false;
        }
        Ok(generated_token)
    }

    async fn inject_input_tokens(
        &mut self,
        _request_id: RequestId,
        _input_token_ids: Vec<u32>,
    ) -> Result<(), InferenceEngineError> {
        Ok(())
    }

    async fn cancel_generation(
        &mut self,
        _request_id: RequestId,
    ) -> Result<GenerationFinalization, InferenceEngineError> {
        self.cancellation_count.fetch_add(1, Ordering::SeqCst);
        self.is_active = false;
        Ok(self.cancelled_generation_finalization)
    }

    async fn update_mlx_memory_limit(
        &mut self,
        requested_mlx_memory_ceiling_bytes: u64,
    ) -> Result<MlxMemoryLimitAdjustment, InferenceEngineError> {
        Ok(MlxMemoryLimitAdjustment::new(
            requested_mlx_memory_ceiling_bytes,
            self.maximum_allocator_cache_memory_limit_bytes
                .min(requested_mlx_memory_ceiling_bytes),
            1,
            ExpertMemoryMode::Resident,
            None,
        )
        .with_expert_residency_telemetry(ExpertResidencyTelemetry {
            total_layer_count: 2,
            resident_expert_count: 2,
            resident_expert_payload_bytes: 8_000,
        }))
    }

    async fn collect_persistent_prompt_cache_stats(
        &self,
    ) -> Result<Option<WorkerEvent>, InferenceEngineError> {
        Ok(self
            .is_active
            .then(|| self.active_generation_prompt_cache_stats.clone())
            .flatten())
    }

    async fn clear_persistent_prompt_cache(
        &mut self,
        _model_id: Option<String>,
    ) -> Result<Option<WorkerEvent>, InferenceEngineError> {
        Ok(self.prompt_cache_clear_event.clone())
    }
}
