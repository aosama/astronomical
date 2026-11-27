//! User-journey coverage for the worker's speech slot: a speech model is
//! swapped in, speaks a request a segment at a time, can be cancelled
//! between segments, refuses what it cannot speak, and finalizes every
//! request.

use crate::FactoryEngines;
use std::collections::VecDeque;
use std::sync::Mutex;

use ipc_protocol::{
    RequestId, SpeechCommand, WorkerCommand, WorkerEvent, WorkerModelConfiguration,
    WorkerSpeechModelConfiguration,
};
use memory_policy::live_memory_limit::{LiveMemoryLimitEngine, LiveMemoryLimitRefusal};
use speech_contract::{SpeechFailure, SpeechRequest};

use super::chat::scripted_chat_test_doubles::{ScriptedChatEngine, ScriptedChatProcessor};
use super::daemon_side::{WorkerUnderTest, start_worker_under_test};
use super::live_memory_limit_journeys::{
    self as journeys, OneMoreRequest, SCRIPTED_FLOOR_BYTES, ScriptedLiveCeiling,
    memory_limit_command,
};
use crate::{
    DecisionUnavailableEngine, EmbeddingUnavailableEngine, EngineBackedWorker,
    ImageGenerationUnavailableEngine, MlxMemoryLimitAdjustment, ModelFactory, ModelFactoryRuntime,
    SpeechEngine, SpeechEngineLoadResult, SpeechEngineStep, StartedSpeech,
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
    FactoryEngines<
        ScriptedChatProcessor,
        ScriptedChatEngine,
        ScriptedSpeechFactory,
        ImageGenerationUnavailableEngine,
        EmbeddingUnavailableEngine,
        DecisionUnavailableEngine,
        ScriptedSpeechEngine,
    >,
>;

/// Speaks every request as two fixed segments.
struct ScriptedSpeechEngine {
    remaining_segments: Option<usize>,
    live_ceiling: ScriptedLiveCeiling,
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

impl LiveMemoryLimitEngine for ScriptedSpeechEngine {
    fn update_mlx_memory_limit(
        &mut self,
        requested_mlx_memory_ceiling_bytes: u64,
    ) -> Result<MlxMemoryLimitAdjustment, LiveMemoryLimitRefusal> {
        // A speech engine moves its ceiling between segments, so a request in
        // progress does not stop it.
        self.live_ceiling.answer(requested_mlx_memory_ceiling_bytes)
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

/// Starts an idle worker, swaps the scripted speech model in, and waits for
/// the swap to be announced.
async fn start_worker_with_a_speech_model() -> (WorkerUnderTest, WorkerEvent) {
    start_worker_with_a_speech_model_holding(None).await
}

/// Like [`start_worker_with_a_speech_model`], for a model that can move its
/// ceiling live down to `live_memory_floor_bytes`.
async fn start_worker_with_a_speech_model_holding(
    live_memory_floor_bytes: Option<u64>,
) -> (WorkerUnderTest, WorkerEvent) {
    let worker: SpeechWorker = EngineBackedWorker::idle_with_model_factory(
        ScriptedSpeechFactory {
            speech_engines: Mutex::new(VecDeque::from([ScriptedSpeechEngine {
                remaining_segments: None,
                live_ceiling: ScriptedLiveCeiling::with_floor(live_memory_floor_bytes),
            }])),
        },
        8_000,
    );
    let mut harness = start_worker_under_test!(worker);
    harness
        .swap_in(
            "/models/speech",
            WorkerModelConfiguration::Speech(WorkerSpeechModelConfiguration {
                model_id: SPEECH_MODEL_ID.to_owned(),
                model_family: "kokoro".to_owned(),
                artifact_revision: "scripted-revision".to_owned(),
            }),
        )
        .await;
    let swap_event = harness.next_event().await;
    (harness, swap_event)
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
    harness.close().await;
}

// Positive: each segment's audio arrives in order, then the result and the
// finalization.
#[tokio::test]
async fn should_send_each_segment_then_complete_and_finalize() {
    let (mut harness, _swap_event) = start_worker_with_a_speech_model().await;

    harness.send(speech_command(71, ROMEO_LINE)).await;

    for expected_segment in 1..=2_u32 {
        let segment_event = harness.next_event().await;
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
    let completed_event = harness.next_event().await;
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
        harness.next_event().await,
        WorkerEvent::SpeechFinalized { request_id, .. } if request_id == RequestId::new(71)
    ));
    harness.close().await;
}

// Edge: a cancel that arrives between segments ends the request there.
#[tokio::test]
async fn should_stop_between_segments_when_cancelled() {
    let (mut harness, _swap_event) = start_worker_with_a_speech_model().await;
    harness.send(speech_command(72, ROMEO_LINE)).await;
    assert!(matches!(
        harness.next_event().await,
        WorkerEvent::SpeechSegmentSpoken {
            completed_segments: 1,
            ..
        }
    ));

    harness
        .send(WorkerCommand::Cancel {
            request_id: RequestId::new(72),
        })
        .await;

    let mut cancelled_event = harness.next_event().await;
    // The second segment may already have been spoken before the cancel was read.
    if matches!(cancelled_event, WorkerEvent::SpeechSegmentSpoken { .. }) {
        cancelled_event = harness.next_event().await;
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
        harness.next_event().await,
        WorkerEvent::SpeechFinalized { request_id, .. } if request_id == RequestId::new(72)
    ));
    harness.close().await;
}

// Negative: blank text is refused at the worker boundary, and a second
// request while one is active is refused as busy without disturbing it.
#[tokio::test]
async fn should_refuse_blank_text_and_a_second_request_while_busy() {
    let (mut harness, _swap_event) = start_worker_with_a_speech_model().await;

    harness.send(speech_command(73, "   ")).await;
    assert!(matches!(
        harness.next_event().await,
        WorkerEvent::SpeechFailed {
            reason: SpeechFailure::InvalidRequest { .. },
            ..
        }
    ));
    assert!(matches!(
        harness.next_event().await,
        WorkerEvent::SpeechFinalized { request_id, .. } if request_id == RequestId::new(73)
    ));

    harness.send(speech_command(74, ROMEO_LINE)).await;
    harness.send(speech_command(75, ROMEO_LINE)).await;
    let mut saw_busy_refusal = false;
    let mut saw_first_request_finish = false;
    while !(saw_busy_refusal && saw_first_request_finish) {
        match harness.next_event().await {
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
    harness.close().await;
}

/// Reads events until the request is finalized, returning how the daemon
/// saw its limit answers and whether the speech completed.
async fn limit_answers_until_finalized(harness: &mut WorkerUnderTest) -> (Vec<WorkerEvent>, bool) {
    let mut limit_answers = Vec::new();
    let mut did_complete = false;
    loop {
        match harness.next_event().await {
            answer @ (WorkerEvent::MlxMemoryLimitChanged { .. }
            | WorkerEvent::MlxMemoryLimitRejected { .. }) => limit_answers.push(answer),
            WorkerEvent::SpeechCompleted { .. } => did_complete = true,
            WorkerEvent::SpeechFinalized { .. } => return (limit_answers, did_complete),
            _ => {}
        }
    }
}

// Positive: a change that arrives between two segments is applied and the
// speech still completes with the same model.
#[tokio::test]
async fn should_apply_a_memory_limit_change_between_segments_and_complete_the_speech() {
    let (mut harness, _swap_event) =
        start_worker_with_a_speech_model_holding(Some(SCRIPTED_FLOOR_BYTES)).await;
    harness.send(speech_command(84, ROMEO_LINE)).await;
    assert!(matches!(
        harness.next_event().await,
        WorkerEvent::SpeechSegmentSpoken {
            completed_segments: 1,
            ..
        }
    ));

    harness.send(memory_limit_command(4_000)).await;

    let (limit_answers, did_complete) = limit_answers_until_finalized(&mut harness).await;
    assert!(did_complete, "the speech must complete after the change");
    assert!(matches!(
        limit_answers.as_slice(),
        [WorkerEvent::MlxMemoryLimitChanged {
            effective_mlx_memory_ceiling_bytes: 4_000,
            ..
        }]
    ));
    harness.close().await;
}

// Edge: a change that arrives right after the last segment, before the
// request is finalized, is answered exactly once.
#[tokio::test]
async fn should_answer_a_memory_limit_change_after_the_last_segment_exactly_once() {
    let (mut harness, _swap_event) =
        start_worker_with_a_speech_model_holding(Some(SCRIPTED_FLOOR_BYTES)).await;
    harness.send(speech_command(85, ROMEO_LINE)).await;
    for _segment in 0..2 {
        assert!(matches!(
            harness.next_event().await,
            WorkerEvent::SpeechSegmentSpoken { .. }
        ));
    }

    harness.send(memory_limit_command(4_000)).await;

    let (limit_answers, did_complete) = limit_answers_until_finalized(&mut harness).await;
    assert!(did_complete);
    assert_eq!(limit_answers.len(), 1, "{limit_answers:?}");
    harness.close().await;
}

fn one_more_speech_request() -> OneMoreRequest {
    OneMoreRequest {
        worker_command: speech_command(81, ROMEO_LINE),
        is_its_completion: is_speech_completion,
    }
}

fn is_speech_completion(worker_event: &WorkerEvent) -> bool {
    matches!(worker_event, WorkerEvent::SpeechCompleted { request_id, .. } if *request_id == RequestId::new(81))
}

#[tokio::test]
async fn should_change_the_memory_limit_live_and_keep_speaking_with_the_same_model() {
    let (harness, _swap_event) =
        start_worker_with_a_speech_model_holding(Some(SCRIPTED_FLOOR_BYTES)).await;
    journeys::assert_a_change_applies_and_the_model_keeps_serving(
        harness,
        one_more_speech_request(),
    )
    .await;
}

#[tokio::test]
async fn should_refuse_a_memory_limit_below_what_the_speech_model_holds_and_keep_speaking() {
    let (harness, _swap_event) =
        start_worker_with_a_speech_model_holding(Some(SCRIPTED_FLOOR_BYTES)).await;
    journeys::assert_a_change_below_the_floor_is_refused_and_the_model_keeps_serving(
        harness,
        one_more_speech_request(),
    )
    .await;
}

#[tokio::test]
async fn should_reject_a_memory_limit_change_cleanly_when_the_speech_engine_cannot_move_it() {
    let (harness, _swap_event) = start_worker_with_a_speech_model().await;
    journeys::assert_an_engine_that_cannot_move_its_ceiling_is_rejected_cleanly(
        harness,
        one_more_speech_request(),
    )
    .await;
}
