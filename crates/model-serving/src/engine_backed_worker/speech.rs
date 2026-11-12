//! Speaks one speech request a segment at a time, sending each segment's
//! audio as it is ready, and publishes the request's cleanup event.
//!
//! Like an image request, a speech request stays active across loop turns:
//! each turn speaks one segment, so a `Cancel` that arrives between
//! segments stops it with at most one segment of wasted work.

use std::time::Instant;

use astronomical_ipc_protocol::{
    MlxMemorySnapshotSource, ProtocolWriter, RequestId, SpeechCommand, SpeechResultMetadata,
    SpokenAudioSegment, WorkerEvent, WorkerModelCapabilities, WorkerSpeechCapabilities,
};
use astronomical_speech_contract::{SpeechFailure, pcm16_bytes};
use tokio::io::AsyncWrite;

use super::support::{ModelFactory, WorkerRuntimeError};
use super::{EngineBackedWorker, LoadedRuntime, output};
use crate::model_generation_processor::ModelGenerationProcessor;
use crate::{
    ImageGenerationEngine, InferenceEngine, SpeechEngineLoadResult, SpeechEngineStep, StartedSpeech,
};

/// The speech request a worker is speaking, and what its result will say.
pub(crate) struct ActiveSpeechGeneration {
    pub(crate) request_id: RequestId,
    started_at: Instant,
    seed: u64,
    started_speech: StartedSpeech,
    sample_count: u64,
    spoken_segment_count: u32,
    cut_short_segment_count: u32,
}

impl ActiveSpeechGeneration {
    fn new(request_id: RequestId, seed: u64, started_speech: StartedSpeech) -> Self {
        Self {
            request_id,
            started_at: Instant::now(),
            seed,
            started_speech,
            sample_count: 0,
            spoken_segment_count: 0,
            cut_short_segment_count: 0,
        }
    }

    pub(crate) fn elapsed_millis(&self) -> u64 {
        u64::try_from(self.started_at.elapsed().as_millis()).unwrap_or(u64::MAX)
    }

    fn record_segment(&mut self, sample_count: usize, was_cut_short: bool) {
        self.sample_count += sample_count as u64;
        self.spoken_segment_count += 1;
        self.cut_short_segment_count += u32::from(was_cut_short);
    }

    fn result_metadata(&self) -> SpeechResultMetadata {
        SpeechResultMetadata {
            sample_rate_hertz: self.started_speech.sample_rate_hertz,
            sample_count: self.sample_count,
            segment_count: self.spoken_segment_count,
            cut_short_segment_count: self.cut_short_segment_count,
            voice_name: self.started_speech.voice_name.clone(),
            seed: self.seed,
        }
    }
}

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
    /// Loads a speech engine and announces its readiness and first memory
    /// sample. Every other runtime is announced by its own arm.
    pub(crate) async fn announce_speech_readiness<WriteTransport>(
        &mut self,
        loaded_runtime: &mut LoadedRuntime<
            Processor,
            Engine,
            ImageEngine,
            EmbeddingsEngine,
            DecisionsEngine,
            TextToSpeechEngine,
        >,
        event_writer: &mut ProtocolWriter<WriteTransport>,
    ) -> Result<(), WorkerRuntimeError>
    where
        WriteTransport: AsyncWrite + Unpin,
    {
        let LoadedRuntime::Speech(speech_engine) = loaded_runtime else {
            return Ok(());
        };
        let speech_load_result = speech_engine.load().map_err(|load_failure| {
            WorkerRuntimeError::InferenceEngineInitializationFailed {
                reason: format!("speech engine initialization failed: {load_failure}"),
            }
        })?;
        self.minimum_mlx_memory_ceiling_bytes = speech_engine.minimum_mlx_memory_ceiling_bytes();
        event_writer
            .send_event(&WorkerEvent::Ready {
                capabilities: speech_model_capabilities(&speech_load_result),
                model_id: speech_load_result.model_id,
            })
            .await?;
        self.emit_mlx_memory_sample(MlxMemorySnapshotSource::ModelLoaded, event_writer)
            .await
    }

    /// Accepts a speech command, or reports why it cannot be spoken.
    ///
    /// The command is checked again here although the daemon already
    /// checked it: the worker does not trust what crosses its boundary.
    pub(crate) async fn start_speech<WriteTransport>(
        &mut self,
        speech_command: SpeechCommand,
        event_writer: &mut ProtocolWriter<WriteTransport>,
    ) -> Result<Option<ActiveSpeechGeneration>, WorkerRuntimeError>
    where
        WriteTransport: AsyncWrite + Unpin,
    {
        let request_id = speech_command.request_id;
        let start_result = match speech_command.validated_request() {
            Err(validation_failure) => Err(validation_failure),
            Ok(speech_request) => match self.loaded_speech_engine() {
                Some(speech_engine) => speech_engine.start_speech(speech_request),
                None => Err(SpeechFailure::fatal_execution(
                    "the loaded model does not speak",
                )),
            },
        };
        match start_result {
            Ok(started_speech) => Ok(Some(ActiveSpeechGeneration::new(
                request_id,
                speech_command.seed,
                started_speech,
            ))),
            Err(start_failure) => {
                self.emit_speech_failure_and_finalization(
                    request_id,
                    start_failure,
                    0,
                    event_writer,
                )
                .await?;
                Ok(None)
            }
        }
    }

    /// Speaks the next segment and sends its audio, or ends the request.
    pub(crate) async fn advance_speech<WriteTransport>(
        &mut self,
        mut active_speech: ActiveSpeechGeneration,
        event_writer: &mut ProtocolWriter<WriteTransport>,
    ) -> Result<Option<ActiveSpeechGeneration>, WorkerRuntimeError>
    where
        WriteTransport: AsyncWrite + Unpin,
    {
        let speech_step = match self.loaded_speech_engine() {
            Some(speech_engine) => speech_engine.advance_speech(),
            None => Err(SpeechFailure::fatal_execution(
                "the loaded speech engine was removed while it was speaking",
            )),
        };
        match speech_step {
            Ok(SpeechEngineStep::SegmentSpoken {
                samples,
                completed_segments,
                total_segments,
                was_cut_short,
            }) => {
                active_speech.record_segment(samples.len(), was_cut_short);
                let segment_event = WorkerEvent::SpeechSegmentSpoken {
                    request_id: active_speech.request_id,
                    audio_segment: SpokenAudioSegment {
                        pcm16_bytes: pcm16_bytes(&samples),
                    },
                    completed_segments: saturating_u32(completed_segments),
                    total_segments: saturating_u32(total_segments),
                    was_cut_short,
                    elapsed_millis: active_speech.elapsed_millis(),
                    mlx_memory_snapshot: self.speech_memory_snapshot(),
                };
                event_writer.send_event(&segment_event).await?;
                Ok(Some(active_speech))
            }
            Ok(SpeechEngineStep::Completed) => {
                self.complete_speech(&active_speech, event_writer).await?;
                Ok(None)
            }
            Err(segment_failure) => {
                self.emit_speech_failure_and_finalization(
                    active_speech.request_id,
                    segment_failure,
                    active_speech.elapsed_millis(),
                    event_writer,
                )
                .await?;
                Ok(None)
            }
        }
    }

    async fn complete_speech<WriteTransport>(
        &mut self,
        active_speech: &ActiveSpeechGeneration,
        event_writer: &mut ProtocolWriter<WriteTransport>,
    ) -> Result<(), WorkerRuntimeError>
    where
        WriteTransport: AsyncWrite + Unpin,
    {
        event_writer
            .send_event(&WorkerEvent::SpeechCompleted {
                request_id: active_speech.request_id,
                result_metadata: active_speech.result_metadata(),
            })
            .await?;
        self.emit_speech_finalization(
            active_speech.request_id,
            active_speech.elapsed_millis(),
            event_writer,
        )
        .await
    }

    /// Stops the active request between segments; its audio so far is dropped.
    pub(crate) async fn cancel_speech<WriteTransport>(
        &mut self,
        active_speech: ActiveSpeechGeneration,
        event_writer: &mut ProtocolWriter<WriteTransport>,
    ) -> Result<(), WorkerRuntimeError>
    where
        WriteTransport: AsyncWrite + Unpin,
    {
        if let Some(speech_engine) = self.loaded_speech_engine() {
            speech_engine.cancel_speech();
        }
        self.emit_speech_failure_and_finalization(
            active_speech.request_id,
            SpeechFailure::Cancelled,
            active_speech.elapsed_millis(),
            event_writer,
        )
        .await
    }

    /// Refuses a speech command that arrived while another request runs.
    pub(crate) async fn refuse_speech_while_busy<WriteTransport>(
        &mut self,
        request_id: RequestId,
        event_writer: &mut ProtocolWriter<WriteTransport>,
    ) -> Result<(), WorkerRuntimeError>
    where
        WriteTransport: AsyncWrite + Unpin,
    {
        event_writer
            .send_event(&WorkerEvent::SpeechFailed {
                request_id,
                reason: SpeechFailure::EngineBusy,
            })
            .await?;
        // The busy request never touched the engine, so its finalization
        // carries no observation of its own.
        event_writer
            .send_event(&WorkerEvent::SpeechFinalized {
                request_id,
                elapsed_millis: 0,
                mlx_memory_snapshot: None,
            })
            .await?;
        Ok(())
    }

    async fn emit_speech_failure_and_finalization<WriteTransport>(
        &mut self,
        request_id: RequestId,
        speech_failure: SpeechFailure,
        elapsed_millis: u64,
        event_writer: &mut ProtocolWriter<WriteTransport>,
    ) -> Result<(), WorkerRuntimeError>
    where
        WriteTransport: AsyncWrite + Unpin,
    {
        event_writer
            .send_event(&WorkerEvent::SpeechFailed {
                request_id,
                reason: speech_failure,
            })
            .await?;
        self.emit_speech_finalization(request_id, elapsed_millis, event_writer)
            .await
    }

    async fn emit_speech_finalization<WriteTransport>(
        &mut self,
        request_id: RequestId,
        elapsed_millis: u64,
        event_writer: &mut ProtocolWriter<WriteTransport>,
    ) -> Result<(), WorkerRuntimeError>
    where
        WriteTransport: AsyncWrite + Unpin,
    {
        // The engine observed MLX after it returned the request's memory, so
        // the finalization publishes the state the next request starts from.
        let mlx_memory_snapshot = self
            .loaded_speech_engine()
            .and_then(|speech_engine| speech_engine.take_post_cleanup_memory_telemetry())
            .map(|mlx_memory_telemetry| {
                output::worker_memory_snapshot(
                    MlxMemorySnapshotSource::Finalized,
                    mlx_memory_telemetry,
                )
            });
        event_writer
            .send_event(&WorkerEvent::SpeechFinalized {
                request_id,
                elapsed_millis,
                mlx_memory_snapshot,
            })
            .await?;
        Ok(())
    }

    fn speech_memory_snapshot(
        &mut self,
    ) -> Option<astronomical_ipc_protocol::WorkerMlxMemorySnapshot> {
        let mlx_memory_telemetry = self
            .loaded_speech_engine()?
            .collect_mlx_memory_telemetry()?;
        Some(output::worker_memory_snapshot(
            MlxMemorySnapshotSource::SpeechSegment,
            mlx_memory_telemetry,
        ))
    }

    fn loaded_speech_engine(&mut self) -> Option<&mut TextToSpeechEngine> {
        match self.loaded_runtime.as_mut() {
            Some(LoadedRuntime::Speech(speech_engine)) => Some(speech_engine),
            Some(LoadedRuntime::Autoregressive(_))
            | Some(LoadedRuntime::Image(_))
            | Some(LoadedRuntime::Embeddings(_))
            | Some(LoadedRuntime::Decisions(_))
            | None => None,
        }
    }
}

/// What a loaded speech model advertises to the daemon.
pub(super) fn speech_model_capabilities(
    speech_load_result: &SpeechEngineLoadResult,
) -> WorkerModelCapabilities {
    WorkerModelCapabilities::speech(WorkerSpeechCapabilities {
        sample_rate_hertz: speech_load_result.sample_rate_hertz,
        voice_names: speech_load_result.voice_names.clone(),
        default_voice_name: speech_load_result.default_voice_name.clone(),
        supports_speed: speech_load_result.supports_speed,
    })
}

/// A segment count as the wire carries it; no request has four billion.
fn saturating_u32(segment_count: usize) -> u32 {
    u32::try_from(segment_count).unwrap_or(u32::MAX)
}
