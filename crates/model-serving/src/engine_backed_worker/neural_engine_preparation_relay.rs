//! Moves Neural Engine programs along while the worker has nothing else to do, and
//! tells the daemon how they are getting on.
//!
//! This happens only between requests, because a step may read a layer's weights on
//! the MLX owner thread, which a generation needs for itself, and because a program's
//! first run wires memory beside it. A few programs already started finish during a
//! generation, and the rest wait for it to end.

use std::time::Duration;

use astronomical_ipc_protocol::{
    NeuralEnginePrefillStatus, ProtocolReader, ProtocolWriter, WorkerCommand, WorkerEvent,
};
use tokio::io::{AsyncRead, AsyncWrite};

use super::support::{ModelFactory, WorkerRuntimeError};
use super::{EngineBackedWorker, LoadedRuntime};
use crate::model_generation_processor::ModelGenerationProcessor;
use crate::{ImageGenerationEngine, InferenceEngine};

/// Often enough that a reused program, which takes about a tenth of a second, waits
/// about as long again for the next one, and rare enough to cost an idle worker
/// nothing it would notice.
const PREPARATION_STEP_INTERVAL: Duration = Duration::from_millis(100);

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
    /// The next command, stepping any preparation along until one arrives. Without
    /// a preparation running this is a plain wait, so an idle worker never wakes.
    pub(super) async fn next_idle_command_preparing_neural_engine<ReadTransport, WriteTransport>(
        &mut self,
        command_reader: &mut ProtocolReader<ReadTransport>,
        event_writer: &mut ProtocolWriter<WriteTransport>,
    ) -> Result<Option<WorkerCommand>, WorkerRuntimeError>
    where
        ReadTransport: AsyncRead + Unpin,
        WriteTransport: AsyncWrite + Unpin,
    {
        loop {
            if let Some(unsent_event) = self.unsent_neural_engine_prefill_event.take() {
                event_writer.send_event(&unsent_event).await?;
            }
            if !self.is_neural_engine_preparation_running {
                if std::mem::take(&mut self.has_request_run_since_settling) {
                    self.report_what_the_last_request_settled(event_writer)
                        .await?;
                }
                return Ok(command_reader.next_command().await?);
            }
            tokio::select! {
                biased;
                next_command = command_reader.next_command() => return Ok(next_command?),
                () = tokio::time::sleep(PREPARATION_STEP_INTERVAL) => {
                    self.step_neural_engine_preparation(event_writer).await?;
                }
            }
        }
    }

    /// A request can find that running programs failed or slow the model down, and they
    /// can only be released with no request active. An error here means the owner thread
    /// is gone, which the next command will report.
    async fn report_what_the_last_request_settled<WriteTransport>(
        &mut self,
        event_writer: &mut ProtocolWriter<WriteTransport>,
    ) -> Result<(), WorkerRuntimeError>
    where
        WriteTransport: AsyncWrite + Unpin,
    {
        let Some(LoadedRuntime::Autoregressive(loaded_model)) = self.loaded_runtime.as_mut() else {
            return Ok(());
        };
        if let Ok(Some(neural_engine_prefill)) = loaded_model
            .engine
            .advance_neural_engine_preparation()
            .await
        {
            event_writer
                .send_event(&WorkerEvent::NeuralEnginePrefillChanged {
                    neural_engine_prefill,
                })
                .await?;
        }
        Ok(())
    }

    async fn step_neural_engine_preparation<WriteTransport>(
        &mut self,
        event_writer: &mut ProtocolWriter<WriteTransport>,
    ) -> Result<(), WorkerRuntimeError>
    where
        WriteTransport: AsyncWrite + Unpin,
    {
        let Some(LoadedRuntime::Autoregressive(loaded_model)) = self.loaded_runtime.as_mut() else {
            self.is_neural_engine_preparation_running = false;
            return Ok(());
        };
        let reported_status = match loaded_model
            .engine
            .advance_neural_engine_preparation()
            .await
        {
            Ok(reported_status) => reported_status,
            // The owner thread is gone, so the programs will never arrive. Saying so
            // ends a count that would otherwise sit in the menu until the next request.
            Err(engine_error) => {
                tracing::warn!(error = %engine_error, "Neural Engine preparation stopped");
                Some(NeuralEnginePrefillStatus::Unavailable {
                    reason: "the Neural Engine preparation stopped".to_owned(),
                })
            }
        };
        let Some(neural_engine_prefill) = reported_status else {
            return Ok(());
        };
        self.is_neural_engine_preparation_running = matches!(
            neural_engine_prefill,
            NeuralEnginePrefillStatus::Preparing { .. }
        );
        event_writer
            .send_event(&WorkerEvent::NeuralEnginePrefillChanged {
                neural_engine_prefill,
            })
            .await?;
        Ok(())
    }
}
