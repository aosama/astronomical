//! Scripted machinery for the worker journeys that cover the bounded
//! re-promotion of a request-demoted streaming engine back to the
//! complete-residency resident engine.
//!
//! The scripted resident engine forks to streaming whenever a request's
//! prompt reaches `RESIDENT_FORK_PROMPT_TOKEN_COUNT`, so each journey
//! controls demotion and the re-promotion hysteresis purely through prompt
//! sizes. The engine role that served every request is recorded so the
//! assertions name which engine actually ran, not just whether output came.

use super::support::worker_model_configuration;
use super::support::{chat_command, close_worker_transport, next_event, ready_event};
use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationFailureReason, ChatGenerationOutput, ChatMessage,
    MAX_IPC_FRAME_BYTES, ProtocolReader, ProtocolWriter, RequestId, WorkerCommand, WorkerEvent,
    WorkerModelConfiguration,
};
use astronomical_model_serving::{
    EngineBackedWorker, EngineGenerationStart, EngineLoadResult, EnginePlacement, GeneratedToken,
    GenerationFinalization, InferenceEngine, InferenceEngineError, ModelFactory,
    ModelFactoryRuntime, ModelGeneratedTokenTranslation, ModelGenerationOutputError,
    ModelGenerationProcessor, PreparedInferenceRequest, PreparedModelGeneration,
    StreamingEngineRequirement, StreamingStartupResidencyFacts, WorkerRuntimeError,
};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::io::{AsyncWrite, duplex, split};
use tokio::task::JoinHandle;
use tokio::time::timeout;

const GIBIBYTE_BYTES: u64 = 1 << 30;
const REQUEST_CONTEXT_SHORTFALL_BYTES: usize = 1 << 30;
const WORKER_EFFECTIVE_CEILING_BYTES: u64 = 64 * GIBIBYTE_BYTES;
const RESIDENT_FORK_PROMPT_TOKEN_COUNT: usize = 100;
pub(super) const DEMOTING_PROMPT_TOKEN_COUNT: usize = 200;
/// Smaller than the demoting prompt, so the hysteresis passes, yet at or
/// above the resident fork threshold, so it still cannot run resident.
pub(super) const MEDIUM_PROMPT_TOKEN_COUNT: usize = 150;
pub(super) const SMALL_PROMPT_TOKEN_COUNT: usize = 10;
/// `chat_command` asks for two output tokens, which every total context adds.
const MAX_OUTPUT_TOKEN_COUNT: usize = 2;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum PromotionEngineRole {
    Resident,
    Streaming,
}

#[derive(Clone, Copy)]
pub(super) struct PromotionScenario {
    pub(super) startup_engine_role: PromotionEngineRole,
    pub(super) complete_residency_fits_effective_ceiling: bool,
    pub(super) resident_promotion_creation_succeeds: bool,
    pub(super) resident_promotion_load_succeeds: bool,
    pub(super) startup_runtime_carries_engine_placement: bool,
}

impl PromotionScenario {
    pub(super) const fn resident_startup_with_room_to_promote() -> Self {
        Self {
            startup_engine_role: PromotionEngineRole::Resident,
            complete_residency_fits_effective_ceiling: true,
            resident_promotion_creation_succeeds: true,
            resident_promotion_load_succeeds: true,
            startup_runtime_carries_engine_placement: true,
        }
    }
}

struct PromotionInferenceRequest {
    prompt_token_count: usize,
}

impl Clone for PromotionInferenceRequest {
    fn clone(&self) -> Self {
        Self {
            prompt_token_count: self.prompt_token_count,
        }
    }
}

impl PreparedInferenceRequest for PromotionInferenceRequest {
    fn prompt_token_count(&self) -> usize {
        self.prompt_token_count
    }

    fn clone_for_streaming_retry(&self) -> Option<Self> {
        Some(self.clone())
    }
}

struct PromotionProcessor;

impl ModelGenerationProcessor for PromotionProcessor {
    type InferenceRequest = PromotionInferenceRequest;
    type RequestOutput = ();

    fn ready_event(&self) -> WorkerEvent {
        ready_event()
    }

    // One prompt character stands for one prompt token, so each journey sets
    // a request's total context exactly through its message length.
    fn prepare_chat_generation(
        &self,
        generation_command: &ChatGenerationCommand,
    ) -> Result<
        PreparedModelGeneration<Self::InferenceRequest, Self::RequestOutput>,
        ChatGenerationFailureReason,
    > {
        let prompt_token_count = generation_command
            .messages
            .iter()
            .map(|chat_message| match chat_message {
                ChatMessage::User { content, .. } => content.chars().count(),
                _ => 0,
            })
            .sum();
        Ok(PreparedModelGeneration::new(
            PromotionInferenceRequest { prompt_token_count },
            (),
        ))
    }

    fn is_end_of_sequence_token(&self, _generated_token_id: u32) -> bool {
        true
    }

    fn translate_generated_token(
        &self,
        _request_output: &mut Self::RequestOutput,
        generated_token_id: u32,
    ) -> Result<ModelGeneratedTokenTranslation, ModelGenerationOutputError> {
        Ok(ModelGeneratedTokenTranslation::from_outputs(vec![
            ChatGenerationOutput::Text {
                text: format!("token-{generated_token_id}"),
            },
        ]))
    }

    fn finish_request_output(
        &self,
        _request_output: &mut Self::RequestOutput,
    ) -> Result<Vec<ChatGenerationOutput>, ModelGenerationOutputError> {
        Ok(Vec::new())
    }
}

struct PromotionEngine {
    role: PromotionEngineRole,
    load_succeeds: bool,
    startup_residency_facts: StreamingStartupResidencyFacts,
    serving_engine_roles: Arc<Mutex<Vec<PromotionEngineRole>>>,
}

impl InferenceEngine for PromotionEngine {
    type Request = PromotionInferenceRequest;

    async fn load(&mut self) -> Result<EngineLoadResult, InferenceEngineError> {
        if !self.load_succeeds {
            return Err(InferenceEngineError::Fatal {
                reason: "the resident engine could not load its weights".to_owned(),
            });
        }
        let engine_load_result = EngineLoadResult::new();
        Ok(match self.role {
            PromotionEngineRole::Resident => engine_load_result,
            PromotionEngineRole::Streaming => {
                engine_load_result.with_startup_residency_facts(Some(self.startup_residency_facts))
            }
        })
    }

    async fn start_generation(
        &mut self,
        inference_request: Self::Request,
    ) -> Result<EngineGenerationStart, InferenceEngineError> {
        if self.role == PromotionEngineRole::Resident
            && inference_request.prompt_token_count >= RESIDENT_FORK_PROMPT_TOKEN_COUNT
        {
            return Err(InferenceEngineError::ResidentForkRequired {
                cause: StreamingEngineRequirement::RequestContextShortfall {
                    shortfall_bytes: REQUEST_CONTEXT_SHORTFALL_BYTES,
                },
                reason: "complete resident experts do not fit the request memory budget".to_owned(),
            });
        }
        self.serving_engine_roles
            .lock()
            .unwrap_or_else(|poisoned_lock| poisoned_lock.into_inner())
            .push(self.role);
        Ok(EngineGenerationStart::new(0))
    }

    async fn decode_next_token(
        &mut self,
        _request_id: RequestId,
    ) -> Result<GeneratedToken, InferenceEngineError> {
        Ok(GeneratedToken::TokenId {
            token_id: 7,
            is_reasoning_token: false,
            expert_memory_mode: None,
            mlx_memory_telemetry: None,
            first_decode_forward_elapsed_millis: None,
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
        Ok(GenerationFinalization::default())
    }
}

#[derive(Clone)]
pub(super) struct PromotionJourneyCounters {
    serving_engine_roles: Arc<Mutex<Vec<PromotionEngineRole>>>,
    streaming_retry_count: Arc<AtomicUsize>,
    resident_promotion_count: Arc<AtomicUsize>,
}

impl PromotionJourneyCounters {
    fn new() -> Self {
        Self {
            serving_engine_roles: Arc::new(Mutex::new(Vec::new())),
            streaming_retry_count: Arc::new(AtomicUsize::new(0)),
            resident_promotion_count: Arc::new(AtomicUsize::new(0)),
        }
    }

    pub(super) fn serving_engine_roles(&self) -> Vec<PromotionEngineRole> {
        self.serving_engine_roles
            .lock()
            .unwrap_or_else(|poisoned_lock| poisoned_lock.into_inner())
            .clone()
    }
}

struct PromotionModelFactory {
    scenario: PromotionScenario,
    counters: PromotionJourneyCounters,
}

impl PromotionModelFactory {
    fn startup_residency_facts(&self) -> StreamingStartupResidencyFacts {
        // Either the complete model comfortably fits the worker ceiling, or
        // its experts alone exceed it, so the fit check has one clear answer.
        let complete_expert_payload_bytes =
            if self.scenario.complete_residency_fits_effective_ceiling {
                4 * GIBIBYTE_BYTES
            } else {
                2 * WORKER_EFFECTIVE_CEILING_BYTES
            };
        let empty_complete_residency_bytes =
            GIBIBYTE_BYTES + complete_expert_payload_bytes + GIBIBYTE_BYTES;
        // Size per-token decoder state so the promotion gate agrees with the
        // scripted resident engine: the largest total context that fits
        // above complete residency is exactly the last one below the fork
        // threshold.
        let largest_resident_total_context_token_count =
            (RESIDENT_FORK_PROMPT_TOKEN_COUNT - 1 + MAX_OUTPUT_TOKEN_COUNT) as u64;
        StreamingStartupResidencyFacts {
            model_core_payload_bytes: GIBIBYTE_BYTES,
            complete_expert_payload_bytes,
            startup_complete_residency_headroom_bytes: GIBIBYTE_BYTES,
            sequence_state_bytes_per_token: WORKER_EFFECTIVE_CEILING_BYTES
                .saturating_sub(empty_complete_residency_bytes)
                / largest_resident_total_context_token_count,
        }
    }

    fn engine(&self, role: PromotionEngineRole) -> PromotionEngine {
        self.engine_with_load_outcome(role, true)
    }

    fn engine_with_load_outcome(
        &self,
        role: PromotionEngineRole,
        load_succeeds: bool,
    ) -> PromotionEngine {
        PromotionEngine {
            role,
            load_succeeds,
            startup_residency_facts: self.startup_residency_facts(),
            serving_engine_roles: Arc::clone(&self.counters.serving_engine_roles),
        }
    }
}

impl ModelFactory<PromotionProcessor, PromotionEngine> for PromotionModelFactory {
    async fn create(
        &self,
        _model_directory: &str,
        _model_configuration: WorkerModelConfiguration,
    ) -> Result<ModelFactoryRuntime<PromotionProcessor, PromotionEngine>, String> {
        if !self.scenario.startup_runtime_carries_engine_placement {
            return Ok(ModelFactoryRuntime::autoregressive_for_streaming_retry(
                PromotionProcessor,
                self.engine(self.scenario.startup_engine_role),
            ));
        }
        let engine_placement = match self.scenario.startup_engine_role {
            PromotionEngineRole::Resident => EnginePlacement::CompleteResidencyFits,
            PromotionEngineRole::Streaming => EnginePlacement::CompleteResidencyDoesNotFit {
                shortfall_bytes: GIBIBYTE_BYTES,
            },
        };
        Ok(ModelFactoryRuntime::autoregressive(
            PromotionProcessor,
            self.engine(self.scenario.startup_engine_role),
            engine_placement,
        ))
    }

    async fn create_streaming_retry(
        &self,
        _model_directory: &str,
        _model_configuration: WorkerModelConfiguration,
    ) -> Result<ModelFactoryRuntime<PromotionProcessor, PromotionEngine>, String> {
        self.counters
            .streaming_retry_count
            .fetch_add(1, Ordering::SeqCst);
        Ok(ModelFactoryRuntime::autoregressive_for_streaming_retry(
            PromotionProcessor,
            self.engine(PromotionEngineRole::Streaming),
        ))
    }

    fn supports_streaming_retry(&self) -> bool {
        true
    }

    fn supports_resident_promotion(&self) -> bool {
        true
    }

    async fn create_resident_promotion(
        &self,
        _model_directory: &str,
        _model_configuration: WorkerModelConfiguration,
    ) -> Result<ModelFactoryRuntime<PromotionProcessor, PromotionEngine>, String> {
        if !self.scenario.resident_promotion_creation_succeeds {
            return Err("the resident engine could not be created".to_owned());
        }
        self.counters
            .resident_promotion_count
            .fetch_add(1, Ordering::SeqCst);
        Ok(ModelFactoryRuntime::autoregressive_for_resident_promotion(
            PromotionProcessor,
            self.engine_with_load_outcome(
                PromotionEngineRole::Resident,
                self.scenario.resident_promotion_load_succeeds,
            ),
        ))
    }
}

pub(super) struct PromotionJourney {
    daemon_reader: ProtocolReader<tokio::io::ReadHalf<tokio::io::DuplexStream>>,
    daemon_writer: ProtocolWriter<tokio::io::WriteHalf<tokio::io::DuplexStream>>,
    pub(super) worker_task: JoinHandle<Result<(), WorkerRuntimeError>>,
    pub(super) counters: PromotionJourneyCounters,
}

impl PromotionJourney {
    pub(super) async fn launch(scenario: PromotionScenario) -> Self {
        let mut promotion_journey = Self::launch_idle(scenario);
        promotion_journey.load_model().await;
        promotion_journey
    }

    pub(super) fn launch_idle(scenario: PromotionScenario) -> Self {
        let counters = PromotionJourneyCounters::new();
        let engine_worker = EngineBackedWorker::idle_with_model_factory(
            PromotionModelFactory {
                scenario,
                counters: counters.clone(),
            },
            WORKER_EFFECTIVE_CEILING_BYTES,
        );
        let (daemon_transport, worker_transport) = duplex(MAX_IPC_FRAME_BYTES * 2);
        let (daemon_reader_transport, daemon_writer_transport) = split(daemon_transport);
        let (worker_reader_transport, worker_writer_transport) = split(worker_transport);
        let worker_task = tokio::spawn(async move {
            engine_worker
                .run(worker_reader_transport, worker_writer_transport)
                .await
        });
        Self {
            daemon_reader: ProtocolReader::new(daemon_reader_transport),
            daemon_writer: ProtocolWriter::new(daemon_writer_transport),
            worker_task,
            counters,
        }
    }

    async fn load_model(&mut self) {
        let model_swap_event = self.request_model_swap().await;
        assert!(
            matches!(model_swap_event, WorkerEvent::ModelSwapped { .. }),
            "the model swap should succeed: {model_swap_event:?}"
        );
    }

    pub(super) async fn request_model_swap(&mut self) -> WorkerEvent {
        assert!(matches!(
            self.next_promotion_event().await,
            WorkerEvent::Idle { .. }
        ));
        send_command(
            &mut self.daemon_writer,
            &WorkerCommand::SwapModel {
                model_directory: "/models/requested-model".to_owned(),
                model_configuration: worker_model_configuration("requested-model"),
            },
        )
        .await;
        self.next_promotion_event().await
    }

    pub(super) async fn require_completed_generation(
        &mut self,
        request_number: u64,
        prompt_token_count: usize,
    ) {
        let mut generation_command = chat_command(request_number, 905);
        generation_command.messages = vec![ChatMessage::User {
            content: "x".repeat(prompt_token_count),
            images: Vec::new(),
        }];
        send_command(
            &mut self.daemon_writer,
            &WorkerCommand::Generate(generation_command),
        )
        .await;
        loop {
            match self.next_promotion_event().await {
                WorkerEvent::Completed { request_id, .. }
                    if request_id == RequestId::new(request_number) =>
                {
                    return;
                }
                WorkerEvent::Failed { request_id, reason }
                    if request_id == RequestId::new(request_number) =>
                {
                    panic!("request {request_number} failed instead of completing: {reason:?}");
                }
                _ => {}
            }
        }
    }

    pub(super) async fn require_failed_generation(
        &mut self,
        request_number: u64,
        prompt_token_count: usize,
    ) -> ChatGenerationFailureReason {
        let mut generation_command = chat_command(request_number, 905);
        generation_command.messages = vec![ChatMessage::User {
            content: "x".repeat(prompt_token_count),
            images: Vec::new(),
        }];
        send_command(
            &mut self.daemon_writer,
            &WorkerCommand::Generate(generation_command),
        )
        .await;
        loop {
            match self.next_promotion_event().await {
                WorkerEvent::Failed { request_id, reason }
                    if request_id == RequestId::new(request_number) =>
                {
                    return reason;
                }
                WorkerEvent::Completed { request_id, .. }
                    if request_id == RequestId::new(request_number) =>
                {
                    panic!("request {request_number} completed instead of failing");
                }
                _ => {}
            }
        }
    }

    pub(super) fn streaming_retry_count(&self) -> usize {
        self.counters.streaming_retry_count.load(Ordering::SeqCst)
    }

    pub(super) fn resident_promotion_count(&self) -> usize {
        self.counters
            .resident_promotion_count
            .load(Ordering::SeqCst)
    }

    async fn next_promotion_event(&mut self) -> WorkerEvent {
        timeout(Duration::from_secs(5), next_event(&mut self.daemon_reader))
            .await
            .expect("the promotion journey should emit an event within five seconds")
    }

    pub(super) async fn close(self) {
        close_worker_transport(self.daemon_writer, self.worker_task).await;
    }
}

async fn send_command<WriteTransport>(
    daemon_writer: &mut ProtocolWriter<WriteTransport>,
    worker_command: &WorkerCommand,
) where
    WriteTransport: AsyncWrite + Unpin,
{
    daemon_writer
        .send_command(worker_command)
        .await
        .expect("the worker should receive the command");
}
