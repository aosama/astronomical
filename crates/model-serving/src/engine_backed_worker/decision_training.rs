//! Training the loaded decision model's final layers, one step per idle
//! loop turn, like speech: the worker stays able to answer commands (and to
//! be cancelled) between steps, and each step's progress goes out as an
//! event so the person sees how far the training has got.

use std::time::Instant;

use astronomical_decision_contract::DecisionFailure;
use astronomical_ipc_protocol::{
    DecisionTrainingCommand, ProtocolReader, ProtocolWriter, RequestId, WorkerEvent,
};
use tokio::io::{AsyncRead, AsyncWrite};

use super::stepped_request::CommandDuringRequest;
use super::support::{ActiveWorkerRequest, ModelFactory, WorkerRuntimeError};
use super::{EngineBackedWorker, LoadedRuntime};
use crate::decision_engine::DecisionTrainingStep;
use crate::model_generation_processor::ModelGenerationProcessor;
use crate::{ImageGenerationEngine, InferenceEngine};

/// A training the worker is running.
pub(crate) struct ActiveDecisionTraining {
    pub(crate) request_id: RequestId,
    started_at: Instant,
}

impl ActiveDecisionTraining {
    fn elapsed_millis(&self) -> u64 {
        u64::try_from(self.started_at.elapsed().as_millis()).unwrap_or(u64::MAX)
    }
}

type ProcessorActiveRequest<Processor> = ActiveWorkerRequest<
    <Processor as ModelGenerationProcessor>::RequestOutput,
    <Processor as ModelGenerationProcessor>::InferenceRequest,
>;

impl<Processor, Engine, Factory, ImageEngine, EmbeddingsEngine, DecisionsEngine, TextToSpeechEngine>
    EngineBackedWorker<
        Processor,
        Engine,
        Factory,
        ImageEngine,
        EmbeddingsEngine,
        DecisionsEngine,
        TextToSpeechEngine,
    >
where
    Processor: ModelGenerationProcessor + Send + 'static,
    Engine: InferenceEngine<Request = Processor::InferenceRequest> + Send + 'static,
    Factory: ModelFactory<
            Processor,
            Engine,
            ImageEngine,
            EmbeddingsEngine,
            DecisionsEngine,
            TextToSpeechEngine,
        > + Send
        + 'static,
    ImageEngine: ImageGenerationEngine,
    EmbeddingsEngine: crate::EmbeddingEngine,
    DecisionsEngine: crate::DecisionEngine,
    TextToSpeechEngine: crate::SpeechEngine,
{
    /// Starts a training on the loaded decision model, or answers why it cannot.
    pub(crate) async fn start_decision_training<WriteTransport>(
        &mut self,
        training_command: DecisionTrainingCommand,
        event_writer: &mut ProtocolWriter<WriteTransport>,
    ) -> Result<Option<ActiveDecisionTraining>, WorkerRuntimeError>
    where
        WriteTransport: AsyncWrite + Unpin,
    {
        let request_id = training_command.request_id;
        let start_outcome = match self.loaded_runtime.as_mut() {
            Some(LoadedRuntime::Decisions(decision_engine)) => {
                decision_engine.start_training(training_command.job)
            }
            Some(_) | None => Err(DecisionFailure::fatal_execution(
                "the loaded model does not answer decision requests",
            )),
        };
        match start_outcome {
            Ok(()) => Ok(Some(ActiveDecisionTraining {
                request_id,
                started_at: Instant::now(),
            })),
            Err(failure_reason) => {
                self.fail_decision_training(request_id, failure_reason, 0, event_writer)
                    .await?;
                Ok(None)
            }
        }
    }

    /// Takes one step; the training stays active until it completes or fails.
    pub(crate) async fn advance_decision_training<WriteTransport>(
        &mut self,
        active_training: ActiveDecisionTraining,
        event_writer: &mut ProtocolWriter<WriteTransport>,
    ) -> Result<Option<ActiveDecisionTraining>, WorkerRuntimeError>
    where
        WriteTransport: AsyncWrite + Unpin,
    {
        let request_id = active_training.request_id;
        let step_started_at = Instant::now();
        let step = match self.loaded_runtime.as_mut() {
            Some(LoadedRuntime::Decisions(decision_engine)) => decision_engine.advance_training(),
            Some(_) | None => Err(DecisionFailure::fatal_execution(
                "the decision model was unloaded during its training",
            )),
        };
        match step {
            Ok(DecisionTrainingStep::Progress {
                stage,
                completed,
                total,
            }) => {
                // How long a step holds the worker is how long a cancel
                // waits, so each one is logged.
                tracing::info!(
                    request_id = ?request_id,
                    ?stage,
                    completed,
                    total,
                    step_millis = step_started_at.elapsed().as_millis() as u64,
                    "decision training step"
                );
                event_writer
                    .send_event(&WorkerEvent::DecisionTrainingProgress {
                        request_id,
                        stage,
                        completed,
                        total,
                    })
                    .await?;
                Ok(Some(active_training))
            }
            Ok(DecisionTrainingStep::Completed(outcome)) => {
                event_writer
                    .send_event(&WorkerEvent::DecisionModelTrained {
                        request_id,
                        outcome,
                        elapsed_millis: active_training.elapsed_millis(),
                    })
                    .await?;
                self.emit_decisions_finalization(
                    request_id,
                    active_training.elapsed_millis(),
                    event_writer,
                )
                .await?;
                Ok(None)
            }
            Err(failure_reason) => {
                let elapsed_millis = active_training.elapsed_millis();
                self.fail_decision_training(
                    request_id,
                    failure_reason,
                    elapsed_millis,
                    event_writer,
                )
                .await?;
                Ok(None)
            }
        }
    }

    /// Trains one step per otherwise idle loop turn; a command that arrives
    /// first is answered without disturbing the training, unless it cancels it.
    pub(crate) async fn serve_active_decision_training<ReadTransport, WriteTransport>(
        &mut self,
        active_training: ActiveDecisionTraining,
        command_reader: &mut ProtocolReader<ReadTransport>,
        event_writer: &mut ProtocolWriter<WriteTransport>,
    ) -> Result<Option<ProcessorActiveRequest<Processor>>, WorkerRuntimeError>
    where
        ReadTransport: AsyncRead + Unpin,
        WriteTransport: AsyncWrite + Unpin,
    {
        tokio::select! {
            biased;
            next_command = command_reader.next_command() => {
                let command_during_request = match next_command? {
                    Some(worker_command) => {
                        self.answer_command_during_request(worker_command, active_training.request_id, event_writer).await?
                    }
                    // The daemon closed the channel; nobody is left to receive the result.
                    None => CommandDuringRequest::Cancel,
                };
                match command_during_request {
                    CommandDuringRequest::Continue => Ok(Some(ActiveWorkerRequest::DecisionTraining(active_training))),
                    CommandDuringRequest::Cancel => {
                        let request_id = active_training.request_id;
                        let elapsed_millis = active_training.elapsed_millis();
                        self.fail_decision_training(
                            request_id,
                            DecisionFailure::invalid_request("the training was cancelled"),
                            elapsed_millis,
                            event_writer,
                        )
                        .await?;
                        Ok(None)
                    }
                }
            }
            () = tokio::task::yield_now() => {
                Ok(self
                    .advance_decision_training(active_training, event_writer)
                    .await?
                    .map(ActiveWorkerRequest::DecisionTraining))
            }
        }
    }

    /// Drops the engine's training state, reports the failure, and finalizes.
    async fn fail_decision_training<WriteTransport>(
        &mut self,
        request_id: RequestId,
        failure_reason: DecisionFailure,
        elapsed_millis: u64,
        event_writer: &mut ProtocolWriter<WriteTransport>,
    ) -> Result<(), WorkerRuntimeError>
    where
        WriteTransport: AsyncWrite + Unpin,
    {
        if let Some(LoadedRuntime::Decisions(decision_engine)) = self.loaded_runtime.as_mut() {
            decision_engine.cancel_training();
        }
        event_writer
            .send_event(&WorkerEvent::DecisionsFailed {
                request_id,
                reason: failure_reason,
            })
            .await?;
        self.emit_decisions_finalization(request_id, elapsed_millis, event_writer)
            .await
    }
}
