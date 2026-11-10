//! Serializes one synchronous decisions request and publishes its cleanup event.

use astronomical_decision_contract::DecisionFailure;
use astronomical_ipc_protocol::{
    DecisionsCommand, MlxMemorySnapshotSource, ProtocolWriter, RequestId, WorkerEvent,
};
use tokio::io::AsyncWrite;

use super::WorkerRuntimeError;
use super::output;
use super::{EngineBackedWorker, LoadedRuntime};
use crate::EmbeddingEngine;
use crate::InferenceEngine;
use crate::engine_backed_worker::support::ModelFactory;
use crate::model_generation_processor::ModelGenerationProcessor;

impl<Processor, Engine, Factory, ImageEngine, EmbeddingsEngine, DecisionsEngine>
    EngineBackedWorker<Processor, Engine, Factory, ImageEngine, EmbeddingsEngine, DecisionsEngine>
where
    Processor: ModelGenerationProcessor + Send + 'static,
    Engine: InferenceEngine<Request = Processor::InferenceRequest> + Send + 'static,
    Factory: ModelFactory<Processor, Engine, ImageEngine, EmbeddingsEngine, DecisionsEngine>
        + Send
        + 'static,
    ImageEngine: crate::ImageGenerationEngine,
    EmbeddingsEngine: EmbeddingEngine,
    DecisionsEngine: crate::DecisionEngine,
{
    /// Answers one decisions command, or reports why it could not be answered.
    ///
    /// The questions are parsed again here although the daemon already
    /// parsed them: the worker does not trust what crosses its boundary.
    pub(crate) async fn start_decisions<WriteTransport>(
        &mut self,
        decisions_command: DecisionsCommand,
        event_writer: &mut ProtocolWriter<WriteTransport>,
    ) -> Result<(), WorkerRuntimeError>
    where
        WriteTransport: AsyncWrite + Unpin,
    {
        let request_id = decisions_command.request_id;
        let started_at = std::time::Instant::now();
        let decide_outcome = match decisions_command.validated_questions() {
            Err(validation_error) => Err(DecisionFailure::invalid_request(
                &validation_error.to_string(),
            )),
            Ok(questions) => match self.loaded_runtime.as_mut() {
                Some(LoadedRuntime::Decisions(decision_engine)) => {
                    decision_engine.decide(&decisions_command.state, &questions)
                }
                Some(_) | None => Err(DecisionFailure::fatal_execution(
                    "the loaded model does not answer decision requests",
                )),
            },
        };
        match decide_outcome {
            Ok(engine_output) => {
                event_writer
                    .send_event(&WorkerEvent::DecisionsCompleted {
                        request_id,
                        question_outcomes: engine_output.question_outcomes,
                        input_token_count: engine_output.input_token_count,
                        elapsed_millis: engine_output.elapsed_millis,
                    })
                    .await?;
            }
            Err(failure_reason) => {
                event_writer
                    .send_event(&WorkerEvent::DecisionsFailed {
                        request_id,
                        reason: failure_reason,
                    })
                    .await?;
            }
        }
        self.emit_decisions_finalization(request_id, elapsed_millis_since(started_at), event_writer)
            .await
    }

    /// Runs a request that answers in one forward pass and leaves nothing
    /// active behind it: a decisions command or an embeddings command.
    pub(crate) async fn start_single_shot_request<WriteTransport>(
        &mut self,
        single_shot_command: astronomical_ipc_protocol::WorkerCommand,
        event_writer: &mut ProtocolWriter<WriteTransport>,
    ) -> Result<(), WorkerRuntimeError>
    where
        WriteTransport: AsyncWrite + Unpin,
    {
        use astronomical_ipc_protocol::WorkerCommand;
        match single_shot_command {
            WorkerCommand::GenerateDecisions(decisions_command) => {
                self.start_decisions(decisions_command, event_writer).await
            }
            WorkerCommand::GenerateEmbeddings(embeddings_command) => {
                self.start_embeddings(embeddings_command, event_writer)
                    .await
            }
            // The idle dispatch routes only those two commands here.
            _ => Ok(()),
        }
    }

    /// Refuses a decisions command that arrived while another request runs.
    pub(crate) async fn refuse_decisions_while_busy<WriteTransport>(
        &mut self,
        request_id: RequestId,
        event_writer: &mut ProtocolWriter<WriteTransport>,
    ) -> Result<(), WorkerRuntimeError>
    where
        WriteTransport: AsyncWrite + Unpin,
    {
        event_writer
            .send_event(&WorkerEvent::DecisionsFailed {
                request_id,
                reason: DecisionFailure::EngineBusy,
            })
            .await?;
        self.emit_decisions_finalization(request_id, 0, event_writer)
            .await
    }

    async fn emit_decisions_finalization<WriteTransport>(
        &mut self,
        request_id: RequestId,
        elapsed_millis: u64,
        event_writer: &mut ProtocolWriter<WriteTransport>,
    ) -> Result<(), WorkerRuntimeError>
    where
        WriteTransport: AsyncWrite + Unpin,
    {
        // The engine captured its post-cleanup observation during the decide
        // call, so the finalization publishes what the status surfaces paint.
        let mlx_memory_snapshot = match self.loaded_runtime.as_mut() {
            Some(LoadedRuntime::Decisions(decision_engine)) => decision_engine
                .take_post_cleanup_memory_telemetry()
                .map(|mlx_memory_telemetry| {
                    output::worker_memory_snapshot(
                        MlxMemorySnapshotSource::Finalized,
                        mlx_memory_telemetry,
                    )
                }),
            _ => None,
        };
        event_writer
            .send_event(&WorkerEvent::DecisionsFinalized {
                request_id,
                elapsed_millis,
                mlx_memory_snapshot,
            })
            .await?;
        Ok(())
    }
}

fn elapsed_millis_since(started_at: std::time::Instant) -> u64 {
    u64::try_from(started_at.elapsed().as_millis()).unwrap_or(u64::MAX)
}
