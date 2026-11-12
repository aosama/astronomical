//! User-journey coverage for the worker's speech slot: a speech model is
//! swapped in, speaks a request a segment at a time, can be cancelled
//! between segments, refuses what it cannot speak, and finalizes every
//! request.

use std::collections::VecDeque;
use std::sync::Mutex;

use astronomical_ipc_protocol::{
    MAX_IPC_FRAME_BYTES, ProtocolReader, ProtocolWriter, RequestId, SpeechCommand, WorkerCommand,
    WorkerEvent, WorkerModelConfiguration, WorkerSpeechModelConfiguration, WorkerSpeechModelFamily,
};
use astronomical_speech_contract::{SpeechFailure, SpeechRequest};
use tokio::io::{duplex, split};
use tokio::task::JoinHandle;

use super::chat::scripted_chat_test_doubles::{ScriptedChatEngine, ScriptedChatProcessor};
use crate::{
    DecisionUnavailableEngine, EmbeddingUnavailableEngine, EngineBackedWorker,
    ImageGenerationUnavailableEngine, ModelFactory, ModelFactoryRuntime, SpeechEngine,
    SpeechEngineLoadResult, SpeechEngineStep, StartedSpeech, WorkerRuntimeError,
};

const SPEECH_MODEL_ID: &str = "scripted-speech-model";
const SEGMENT_SAMPLES: [f32; 2] = [0.0, 0.5];
const ROMEO_LINE: &str = "But soft, what light through yonder window breaks?";

type ScriptedSpeechRuntime = ModelFactoryRuntime<
    ScriptedChatProcessor,
    ScriptedChatEngine,
    ImageGenerationUnavailableEngine,
    EmbeddingUnavailableEngine,
    DecisionUnavailableEngine,
    ScriptedSpeechEngine,
>;

type SpeechWorker = EngineBackedWorker<
    ScriptedChatProcessor,
    ScriptedChatEngine,
    ScriptedSpeechFactory,
    ImageGenerationUnavailableEngine,
    EmbeddingUnavailableEngine,
    DecisionUnavailableEngine,
    ScriptedSpeechEngine,
>;

type DaemonReader = ProtocolReader<tokio::io::ReadHalf<tokio::io::DuplexStream>>;
type DaemonWriter = ProtocolWriter<tokio::io::WriteHalf<tokio::io::DuplexStream>>;

/// Speaks every request as two fixed segments.
struct ScriptedSpeechEngine {
    remaining_segments: Option<usize>,
}

impl SpeechEngine for ScriptedSpeechEngine {
    fn load(&mut self) -> Result<SpeechEngineLoadResult, SpeechFailure> {
        Ok(SpeechEngineLoadResult {
            model_id: SPEECH_MODEL_ID.to_owned(),
            sample_rate_hertz: 24_000,
            voice_names: vec!["af_heart".to_owned()],
            default_voice_name: "af_heart".to_owned(),
            supports_speed: true,
        })
    }

    fn start_speech(
        &mut self,
        speech_request: SpeechRequest,
    ) -> Result<StartedSpeech, SpeechFailure> {
        if self.remaining_segments.is_some() {
            return Err(SpeechFailure::EngineBusy);
        }
        self.remaining_segments = Some(2);
        Ok(StartedSpeech {
            voice_name: speech_request
                .voice_name
                .unwrap_or_else(|| "af_heart".to_owned()),
            sample_rate_hertz: 24_000,
            total_segments: 2,
        })
    }

    fn advance_speech(&mut self) -> Result<SpeechEngineStep, SpeechFailure> {
        let remaining_segments = self
            .remaining_segments
            .ok_or_else(|| SpeechFailure::fatal_execution("nothing to speak"))?;
        if remaining_segments == 0 {
            self.remaining_segments = None;
            return Ok(SpeechEngineStep::Completed);
        }
        self.remaining_segments = Some(remaining_segments - 1);
        Ok(SpeechEngineStep::SegmentSpoken {
            samples: SEGMENT_SAMPLES.to_vec(),
            completed_segments: 3 - remaining_segments,
            total_segments: 2,
            was_cut_short: false,
        })
    }

    fn cancel_speech(&mut self) {
        self.remaining_segments = None;
    }
}

struct ScriptedSpeechFactory {
    speech_engines: Mutex<VecDeque<ScriptedSpeechEngine>>,
}

impl
    ModelFactory<
        ScriptedChatProcessor,
        ScriptedChatEngine,
        ImageGenerationUnavailableEngine,
        EmbeddingUnavailableEngine,
        DecisionUnavailableEngine,
        ScriptedSpeechEngine,
    > for ScriptedSpeechFactory
{
    async fn create(
        &self,
        _model_directory: &str,
        _model_configuration: WorkerModelConfiguration,
    ) -> Result<ScriptedSpeechRuntime, String> {
        let speech_engine = self
            .speech_engines
            .lock()
            .expect("the scripted engine queue should not be poisoned")
            .pop_front()
            .expect("the test should configure every model creation");
        Ok(ModelFactoryRuntime::Speech(speech_engine))
    }
}

struct SpeechWorkerHarness {
    events: DaemonReader,
    commands: DaemonWriter,
    worker_task: JoinHandle<Result<(), WorkerRuntimeError>>,
}

/// Starts an idle worker, swaps the scripted speech model in, and waits for
/// the swap to be announced.
async fn start_worker_with_a_speech_model() -> (SpeechWorkerHarness, WorkerEvent) {
    let worker: SpeechWorker = EngineBackedWorker::idle_with_model_factory(
        ScriptedSpeechFactory {
            speech_engines: Mutex::new(VecDeque::from([ScriptedSpeechEngine {
                remaining_segments: None,
            }])),
        },
        8_000,
    );
    let (daemon_transport, worker_transport) = duplex(MAX_IPC_FRAME_BYTES * 2);
    let (daemon_read, daemon_write) = split(daemon_transport);
    let (worker_read, worker_write) = split(worker_transport);
    let worker_task = tokio::spawn(async move { worker.run(worker_read, worker_write).await });
    let mut harness = SpeechWorkerHarness {
        events: ProtocolReader::new(daemon_read),
        commands: ProtocolWriter::new(daemon_write),
        worker_task,
    };
    assert!(matches!(
        next_event(&mut harness.events).await,
        WorkerEvent::Idle { .. }
    ));
    send(
        &mut harness,
        WorkerCommand::SwapModel {
            model_directory: "/models/speech".to_owned(),
            model_configuration: WorkerModelConfiguration::Speech(WorkerSpeechModelConfiguration {
                model_id: SPEECH_MODEL_ID.to_owned(),
                model_family: WorkerSpeechModelFamily::Kokoro,
                artifact_revision: "scripted-revision".to_owned(),
            }),
        },
    )
    .await;
    let swap_event = next_event(&mut harness.events).await;
    (harness, swap_event)
}

/// The next event that is not a periodic memory sample.
async fn next_event(events: &mut DaemonReader) -> WorkerEvent {
    loop {
        let worker_event = super::next_event_skipping_load_progress(events).await;
        if !matches!(worker_event, WorkerEvent::MlxMemorySample { .. }) {
            return worker_event;
        }
    }
}

async fn send(harness: &mut SpeechWorkerHarness, worker_command: WorkerCommand) {
    harness
        .commands
        .send_command(&worker_command)
        .await
        .expect("the command should reach the worker");
}

fn speech_command(request_number: u64, text: &str) -> WorkerCommand {
    WorkerCommand::GenerateSpeech(SpeechCommand {
        request_id: RequestId::new(request_number),
        model: SPEECH_MODEL_ID.to_owned(),
        text: text.to_owned(),
        voice_name: None,
        speed: 1.0,
        seed: 42,
    })
}

async fn close(harness: SpeechWorkerHarness) {
    harness
        .commands
        .close()
        .await
        .expect("the command transport should close");
    harness
        .worker_task
        .await
        .expect("the worker task should not panic")
        .expect("the worker should stop cleanly when its transport closes");
}

// Positive: a speech model swaps in and advertises its voices.
#[tokio::test]
async fn should_swap_in_a_speech_model_and_advertise_its_voices() {
    let (harness, swap_event) = start_worker_with_a_speech_model().await;

    let WorkerEvent::ModelSwapped {
        model_id,
        capabilities,
        ..
    } = swap_event
    else {
        panic!("the worker should announce the swapped speech model: {swap_event:?}");
    };
    assert_eq!(model_id, SPEECH_MODEL_ID);
    let speech_capabilities = capabilities
        .speech
        .expect("a speech model advertises its speech surface");
    assert_eq!(speech_capabilities.voice_names, ["af_heart"]);
    assert!(capabilities.chat.is_none() && capabilities.decisions.is_none());
    close(harness).await;
}

// Positive: each segment's audio arrives in order, then the result and the
// finalization.
#[tokio::test]
async fn should_send_each_segment_then_complete_and_finalize() {
    let (mut harness, _swap_event) = start_worker_with_a_speech_model().await;

    send(&mut harness, speech_command(71, ROMEO_LINE)).await;

    for expected_segment in 1..=2_u32 {
        let segment_event = next_event(&mut harness.events).await;
        let WorkerEvent::SpeechSegmentSpoken {
            audio_segment,
            completed_segments,
            ..
        } = segment_event
        else {
            panic!("the worker should send segment {expected_segment}: {segment_event:?}");
        };
        assert_eq!(completed_segments, expected_segment);
        assert_eq!(audio_segment.pcm16_bytes, [0x00, 0x00, 0x00, 0x40]);
    }
    let completed_event = next_event(&mut harness.events).await;
    let WorkerEvent::SpeechCompleted {
        result_metadata, ..
    } = completed_event
    else {
        panic!("the worker should complete the request: {completed_event:?}");
    };
    assert_eq!(result_metadata.sample_count, 4);
    assert_eq!(result_metadata.segment_count, 2);
    assert_eq!(result_metadata.seed, 42);
    assert!(matches!(
        next_event(&mut harness.events).await,
        WorkerEvent::SpeechFinalized { request_id, .. } if request_id == RequestId::new(71)
    ));
    close(harness).await;
}

// Edge: a cancel that arrives between segments ends the request there.
#[tokio::test]
async fn should_stop_between_segments_when_cancelled() {
    let (mut harness, _swap_event) = start_worker_with_a_speech_model().await;
    send(&mut harness, speech_command(72, ROMEO_LINE)).await;
    assert!(matches!(
        next_event(&mut harness.events).await,
        WorkerEvent::SpeechSegmentSpoken {
            completed_segments: 1,
            ..
        }
    ));

    send(
        &mut harness,
        WorkerCommand::Cancel {
            request_id: RequestId::new(72),
        },
    )
    .await;

    let mut cancelled_event = next_event(&mut harness.events).await;
    // The second segment may already have been spoken before the cancel was read.
    if matches!(cancelled_event, WorkerEvent::SpeechSegmentSpoken { .. }) {
        cancelled_event = next_event(&mut harness.events).await;
    }
    assert!(
        matches!(
            cancelled_event,
            WorkerEvent::SpeechFailed {
                reason: SpeechFailure::Cancelled,
                ..
            } | WorkerEvent::SpeechCompleted { .. }
        ),
        "{cancelled_event:?}"
    );
    assert!(matches!(
        next_event(&mut harness.events).await,
        WorkerEvent::SpeechFinalized { request_id, .. } if request_id == RequestId::new(72)
    ));
    close(harness).await;
}

// Negative: blank text is refused at the worker boundary, and a second
// request while one is active is refused as busy without disturbing it.
#[tokio::test]
async fn should_refuse_blank_text_and_a_second_request_while_busy() {
    let (mut harness, _swap_event) = start_worker_with_a_speech_model().await;

    send(&mut harness, speech_command(73, "   ")).await;
    assert!(matches!(
        next_event(&mut harness.events).await,
        WorkerEvent::SpeechFailed {
            reason: SpeechFailure::InvalidRequest { .. },
            ..
        }
    ));
    assert!(matches!(
        next_event(&mut harness.events).await,
        WorkerEvent::SpeechFinalized { request_id, .. } if request_id == RequestId::new(73)
    ));

    send(&mut harness, speech_command(74, ROMEO_LINE)).await;
    send(&mut harness, speech_command(75, ROMEO_LINE)).await;
    let mut saw_busy_refusal = false;
    let mut saw_first_request_finish = false;
    while !(saw_busy_refusal && saw_first_request_finish) {
        match next_event(&mut harness.events).await {
            WorkerEvent::SpeechFailed {
                request_id,
                reason: SpeechFailure::EngineBusy,
            } if request_id == RequestId::new(75) => saw_busy_refusal = true,
            WorkerEvent::SpeechFinalized { request_id, .. } if request_id == RequestId::new(74) => {
                saw_first_request_finish = true;
            }
            _ => {}
        }
    }
    close(harness).await;
}
