use super::*;

pub(super) struct ScriptedChatProcessor {
    emits_parallel_tool_calls: bool,
    prompt_token_count: usize,
}

pub(super) struct LazyScriptedModelFactory {
    pub(super) model_factory_call_count: Arc<AtomicUsize>,
    pub(super) mlx_memory_ceiling_bytes: Arc<AtomicU64>,
}

pub(super) struct FirstCreationFailsScriptedModelFactory {
    pub(super) model_factory_call_count: Arc<AtomicUsize>,
}

impl ModelFactory<ScriptedChatProcessor, ScriptedChatEngine> for LazyScriptedModelFactory {
    async fn create(
        &self,
        _model_directory: &str,
        _max_output_tokens: u32,
    ) -> Result<(ScriptedChatProcessor, ScriptedChatEngine), String> {
        self.model_factory_call_count.fetch_add(1, Ordering::SeqCst);
        Ok((ScriptedChatProcessor::new(), ScriptedChatEngine::new()))
    }

    fn update_mlx_memory_ceiling_bytes(&mut self, effective_mlx_memory_ceiling_bytes: u64) {
        self.mlx_memory_ceiling_bytes
            .store(effective_mlx_memory_ceiling_bytes, Ordering::SeqCst);
    }
}

impl ModelFactory<ScriptedChatProcessor, ScriptedChatEngine>
    for FirstCreationFailsScriptedModelFactory
{
    async fn create(
        &self,
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

pub(super) struct MalformedFinishProcessor;
pub(super) struct CorrectionRequestingProcessor;

impl ScriptedChatProcessor {
    pub(super) fn new() -> Self {
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
    ) -> WorkerEvent {
        ready_event_with_load_details(mtp_runtime_state, mtp_unavailable_reason)
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
    ) -> WorkerEvent {
        ready_event_with_load_details(mtp_runtime_state, mtp_unavailable_reason)
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
    ) -> WorkerEvent {
        ready_event_with_load_details(mtp_runtime_state, mtp_unavailable_reason)
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

pub(super) struct ScriptedChatEngine {
    cancellation_count: Arc<AtomicUsize>,
    cached_token_count: u32,
    generated_tokens: Vec<GeneratedToken>,
    is_active: bool,
    next_token_index: usize,
    fatal_decode_reason: Option<String>,
    pub(super) initial_expert_memory_mode: Option<ExpertMemoryMode>,
    cancelled_generation_finalization: GenerationFinalization,
    mtp_runtime_state: MtpRuntimeState,
    mtp_unavailable_reason: Option<String>,
}

impl ScriptedChatEngine {
    pub(super) fn new() -> Self {
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
                    generation_finalization: None,
                },
                GeneratedToken::TokenId {
                    token_id: 2,
                    is_reasoning_token: false,
                    expert_memory_mode: None,
                    mlx_memory_telemetry: None,
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
            generated_tokens,
            is_active: false,
            next_token_index: 0,
            fatal_decode_reason: None,
            initial_expert_memory_mode: None,
            cancelled_generation_finalization: GenerationFinalization::default(),
            mtp_runtime_state: MtpRuntimeState::Disabled,
            mtp_unavailable_reason: None,
        }
    }

    pub(super) fn with_fatal_decode_reason(fatal_decode_reason: &str) -> Self {
        Self {
            cancellation_count: Arc::new(AtomicUsize::new(0)),
            cached_token_count: 0,
            generated_tokens: Vec::new(),
            is_active: false,
            next_token_index: 0,
            fatal_decode_reason: Some(fatal_decode_reason.to_owned()),
            initial_expert_memory_mode: None,
            cancelled_generation_finalization: GenerationFinalization::default(),
            mtp_runtime_state: MtpRuntimeState::Disabled,
            mtp_unavailable_reason: None,
        }
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
                },
            )),
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

    pub(super) fn cancellation_count(&self) -> Arc<AtomicUsize> {
        Arc::clone(&self.cancellation_count)
    }
}

impl InferenceEngine for ScriptedChatEngine {
    type Request = ScriptedInferenceRequest;

    async fn load(&mut self) -> Result<EngineLoadResult, InferenceEngineError> {
        let mut engine_load_result =
            EngineLoadResult::new().with_mtp_runtime_state(self.mtp_runtime_state);
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
        Ok(self.initial_expert_memory_mode.map_or_else(
            || EngineGenerationStart::new(self.cached_token_count),
            |expert_memory_mode| {
                EngineGenerationStart::with_expert_memory_mode(
                    self.cached_token_count,
                    expert_memory_mode,
                )
            },
        ))
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
            .copied()
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
            1,
            ExpertMemoryMode::Resident,
            None,
        ))
    }
}

pub(super) struct TrackingChatEngine {
    is_active: bool,
}

impl TrackingChatEngine {
    pub(super) fn new() -> Self {
        Self { is_active: false }
    }
}

impl InferenceEngine for TrackingChatEngine {
    type Request = ScriptedInferenceRequest;

    async fn load(&mut self) -> Result<EngineLoadResult, InferenceEngineError> {
        Ok(EngineLoadResult::new())
    }

    async fn start_generation(
        &mut self,
        _generation_request: ScriptedInferenceRequest,
    ) -> Result<EngineGenerationStart, InferenceEngineError> {
        if self.is_active {
            return Err(InferenceEngineError::EngineBusy);
        }
        self.is_active = true;
        Ok(EngineGenerationStart::new(0))
    }

    async fn decode_next_token(
        &mut self,
        _request_id: RequestId,
    ) -> Result<GeneratedToken, InferenceEngineError> {
        Ok(GeneratedToken::TokenId {
            token_id: 1,
            is_reasoning_token: false,
            expert_memory_mode: None,
            mlx_memory_telemetry: None,
            generation_finalization: None,
        })
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
        self.is_active = false;
        Ok(GenerationFinalization::default())
    }
}
