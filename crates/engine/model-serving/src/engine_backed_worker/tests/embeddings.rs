//! User-journey coverage for the worker's embeddings slot with a scripted
//! engine: a live memory-limit change is applied or refused while the same
//! embedding model stays loaded and keeps serving.

use crate::FactoryEngines;
use std::collections::VecDeque;
use std::sync::Mutex;

use ipc_protocol::{
    EmbeddingEncodingFormat, EmbeddingsCommand, EmbeddingsFailureReason, RequestId, WorkerCommand,
    WorkerEmbeddingCapabilities, WorkerEmbeddingModelConfiguration, WorkerEvent,
    WorkerModelConfiguration,
};
use memory_policy::live_memory_limit::{LiveMemoryLimitEngine, LiveMemoryLimitRefusal};

use super::chat::scripted_chat_test_doubles::{ScriptedChatEngine, ScriptedChatProcessor};
use super::daemon_side::{WorkerUnderTest, start_worker_under_test};
use super::live_memory_limit_journeys::{
    self as journeys, OneMoreRequest, SCRIPTED_FLOOR_BYTES, ScriptedLiveCeiling,
};
use crate::{
    DecisionUnavailableEngine, EmbeddingEngine, EmbeddingEngineLoadResult, EmbeddingEngineOutput,
    EngineBackedWorker, ImageGenerationUnavailableEngine, MlxMemoryLimitAdjustment, ModelFactory,
    ModelFactoryRuntime, SpeechUnavailableEngine,
};

const EMBEDDING_MODEL_ID: &str = "scripted-embedding-model";
const VECTOR_WIDTH: u32 = 4;
const SONNET_LINE: &str = "Shall I compare thee to a summer's day?";

type ScriptedEmbeddingRuntime = ModelFactoryRuntime<
    ScriptedChatProcessor,
    ScriptedChatEngine,
    ImageGenerationUnavailableEngine,
    ScriptedEmbeddingEngine,
    DecisionUnavailableEngine,
    SpeechUnavailableEngine,
>;

type EmbeddingsWorker = EngineBackedWorker<
    FactoryEngines<
        ScriptedChatProcessor,
        ScriptedChatEngine,
        ScriptedEmbeddingFactory,
        ImageGenerationUnavailableEngine,
        ScriptedEmbeddingEngine,
        DecisionUnavailableEngine,
        SpeechUnavailableEngine,
    >,
>;

/// Embeds every input as the same unit vector.
struct ScriptedEmbeddingEngine {
    live_ceiling: ScriptedLiveCeiling,
}

impl EmbeddingEngine for ScriptedEmbeddingEngine {
    fn load(&mut self) -> Result<EmbeddingEngineLoadResult, EmbeddingsFailureReason> {
        Ok(EmbeddingEngineLoadResult::new(
            EMBEDDING_MODEL_ID,
            WorkerEmbeddingCapabilities {
                vector_width: VECTOR_WIDTH,
                max_input_tokens: 512,
            },
        ))
    }

    fn embed(
        &mut self,
        embeddings_command: &EmbeddingsCommand,
    ) -> Result<EmbeddingEngineOutput, EmbeddingsFailureReason> {
        Ok(EmbeddingEngineOutput {
            embeddings: vec![vec![0.5; VECTOR_WIDTH as usize]; embeddings_command.inputs.len()],
            input_token_counts: vec![9; embeddings_command.inputs.len()],
            elapsed_millis: 2,
        })
    }
}

impl LiveMemoryLimitEngine for ScriptedEmbeddingEngine {
    fn update_mlx_memory_limit(
        &mut self,
        requested_mlx_memory_ceiling_bytes: u64,
    ) -> Result<MlxMemoryLimitAdjustment, LiveMemoryLimitRefusal> {
        self.live_ceiling.answer(requested_mlx_memory_ceiling_bytes)
    }
}

struct ScriptedEmbeddingFactory {
    embedding_engines: Mutex<VecDeque<ScriptedEmbeddingEngine>>,
}

impl
    ModelFactory<
        ScriptedChatProcessor,
        ScriptedChatEngine,
        ImageGenerationUnavailableEngine,
        ScriptedEmbeddingEngine,
        DecisionUnavailableEngine,
        SpeechUnavailableEngine,
    > for ScriptedEmbeddingFactory
{
    async fn create(
        &self,
        _model_directory: &str,
        _model_configuration: WorkerModelConfiguration,
    ) -> Result<ScriptedEmbeddingRuntime, String> {
        let embedding_engine = self
            .embedding_engines
            .lock()
            .expect("the scripted engine queue should not be poisoned")
            .pop_front()
            .expect("the test should configure every model creation");
        Ok(ModelFactoryRuntime::Embeddings(embedding_engine))
    }
}

/// Starts an idle worker, swaps the scripted embedding model in, and waits
/// for the swap to be announced.
async fn start_worker_with_an_embedding_model_holding(
    live_memory_floor_bytes: Option<u64>,
) -> WorkerUnderTest {
    let worker: EmbeddingsWorker = EngineBackedWorker::idle_with_model_factory(
        ScriptedEmbeddingFactory {
            embedding_engines: Mutex::new(VecDeque::from([ScriptedEmbeddingEngine {
                live_ceiling: ScriptedLiveCeiling::with_floor(live_memory_floor_bytes),
            }])),
        },
        8_000,
    );
    let mut worker_under_test = start_worker_under_test!(worker);
    worker_under_test
        .swap_in(
            "/models/embedding",
            WorkerModelConfiguration::Embeddings(WorkerEmbeddingModelConfiguration {
                model_id: EMBEDDING_MODEL_ID.to_owned(),
                model_family: "modernbert".to_owned(),
                artifact_revision: "scripted-revision".to_owned(),
                vector_width: VECTOR_WIDTH,
                maximum_input_tokens: 512,
            }),
        )
        .await;
    assert!(matches!(
        worker_under_test.next_event().await,
        WorkerEvent::ModelSwapped { .. }
    ));
    worker_under_test
}

fn one_more_embeddings_request() -> OneMoreRequest {
    OneMoreRequest {
        worker_command: WorkerCommand::GenerateEmbeddings(EmbeddingsCommand {
            request_id: RequestId::new(101),
            model: EMBEDDING_MODEL_ID.to_owned(),
            inputs: vec![SONNET_LINE.to_owned()],
            encoding_format: EmbeddingEncodingFormat::Float,
            dimensions: None,
        }),
        is_its_completion: is_embeddings_completion,
    }
}

fn is_embeddings_completion(worker_event: &WorkerEvent) -> bool {
    matches!(worker_event, WorkerEvent::EmbeddingsCompleted { request_id, .. } if *request_id == RequestId::new(101))
}

#[tokio::test]
async fn should_change_the_memory_limit_live_and_keep_embedding_with_the_same_model() {
    journeys::assert_a_change_applies_and_the_model_keeps_serving(
        start_worker_with_an_embedding_model_holding(Some(SCRIPTED_FLOOR_BYTES)).await,
        one_more_embeddings_request(),
    )
    .await;
}

#[tokio::test]
async fn should_refuse_a_memory_limit_below_what_the_embedding_model_holds_and_keep_embedding() {
    journeys::assert_a_change_below_the_floor_is_refused_and_the_model_keeps_serving(
        start_worker_with_an_embedding_model_holding(Some(SCRIPTED_FLOOR_BYTES)).await,
        one_more_embeddings_request(),
    )
    .await;
}

#[tokio::test]
async fn should_reject_a_memory_limit_change_cleanly_when_the_embedding_engine_cannot_move_it() {
    journeys::assert_an_engine_that_cannot_move_its_ceiling_is_rejected_cleanly(
        start_worker_with_an_embedding_model_holding(None).await,
        one_more_embeddings_request(),
    )
    .await;
}
