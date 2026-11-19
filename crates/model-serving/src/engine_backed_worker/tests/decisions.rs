//! User-journey coverage for the worker's decisions slot: a decision model
//! is swapped in, answers requests, reports failures as request outcomes,
//! and finalizes every request with its memory observation.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use decision_contract::{
    DecisionAnswer, DecisionFailure, DecisionQuestion, FloatVector, HeadInputRequest, HeadInputs,
    NoulAnswer, OrderedJsonValue, QuestionInputCoverage, QuestionOutcome,
};
use ipc_protocol::{
    DecisionsCommand, MAX_IPC_FRAME_BYTES, MlxMemorySnapshotSource, ProtocolReader, ProtocolWriter,
    RequestId, WorkerCommand, WorkerDecisionModelConfiguration, WorkerDecisionModelFamily,
    WorkerEvent, WorkerModelConfiguration,
};
use tokio::io::{duplex, split};
use tokio::task::JoinHandle;

use super::chat::scripted_chat_test_doubles::{ScriptedChatEngine, ScriptedChatProcessor};
use crate::decision_engine::DecisionTrainingStep;
use crate::{
    DecisionEngine, DecisionEngineLoadResult, DecisionEngineOutput, EmbeddingUnavailableEngine,
    EngineBackedWorker, ImageGenerationUnavailableEngine, MlxActiveMemoryBreakdown,
    MlxMemoryTelemetry, ModelFactory, ModelFactoryRuntime, WorkerRuntimeError,
};
use decision_contract::decision_training_job::DecisionTrainingJob;

pub(super) const DECISION_MODEL_ID: &str = "scripted-decision-model";
const MAXIMUM_TOKENS_PER_QUESTION: u32 = 512;
const HAMLET_LINE: &str = "To be, or not to be, that is the question.";

type ScriptedDecisionRuntime = ModelFactoryRuntime<
    ScriptedChatProcessor,
    ScriptedChatEngine,
    ImageGenerationUnavailableEngine,
    EmbeddingUnavailableEngine,
    ScriptedDecisionEngine,
>;

type DecisionsWorker = EngineBackedWorker<
    ScriptedChatProcessor,
    ScriptedChatEngine,
    ScriptedDecisionFactory,
    ImageGenerationUnavailableEngine,
    EmbeddingUnavailableEngine,
    ScriptedDecisionEngine,
>;

type DaemonReader = ProtocolReader<tokio::io::ReadHalf<tokio::io::DuplexStream>>;
type DaemonWriter = ProtocolWriter<tokio::io::WriteHalf<tokio::io::DuplexStream>>;

/// Answers each request with the next scripted outcome and records the
/// question names it was asked, so a test can see what crossed the boundary.
pub(super) struct ScriptedDecisionEngine {
    loaded_model_id: String,
    scripted_outcomes: VecDeque<Result<(), DecisionFailure>>,
    asked_question_names: Arc<Mutex<Vec<String>>>,
    has_unreported_cleanup: bool,
    /// One step per `advance_training`; a model with none scripted
    /// refuses to train, as an untrainable model does.
    scripted_training_steps: VecDeque<Result<DecisionTrainingStep, DecisionFailure>>,
    /// Whether the worker dropped the training state after a failure.
    was_training_cancelled: Arc<Mutex<bool>>,
}

impl DecisionEngine for ScriptedDecisionEngine {
    fn load(&mut self) -> Result<DecisionEngineLoadResult, DecisionFailure> {
        Ok(DecisionEngineLoadResult {
            model_id: self.loaded_model_id.clone(),
            maximum_tokens_per_question: MAXIMUM_TOKENS_PER_QUESTION,
        })
    }

    fn decide(
        &mut self,
        _state: &OrderedJsonValue,
        questions: &[DecisionQuestion],
        head_input_request: HeadInputRequest,
    ) -> Result<DecisionEngineOutput, DecisionFailure> {
        self.has_unreported_cleanup = true;
        let scripted_outcome = self
            .scripted_outcomes
            .pop_front()
            .expect("the test should script every decide call");
        scripted_outcome?;
        let mut question_outcomes = Vec::with_capacity(questions.len());
        for question in questions {
            self.asked_question_names
                .lock()
                .expect("the asked-question log should not be poisoned")
                .push(question.name.clone());
            let mut question_outcome = even_noul_outcome(&question.name);
            if head_input_request == HeadInputRequest::Include {
                question_outcome.head_inputs = Some(scripted_head_inputs());
            }
            question_outcomes.push(question_outcome);
        }
        Ok(DecisionEngineOutput {
            question_outcomes,
            input_token_count: 24,
            elapsed_millis: 3,
        })
    }

    fn start_training(&mut self, job: DecisionTrainingJob) -> Result<(), DecisionFailure> {
        let _ = job;
        if self.scripted_training_steps.is_empty() {
            return Err(DecisionFailure::invalid_request(
                "this model's final layers cannot be trained",
            ));
        }
        Ok(())
    }

    fn advance_training(&mut self) -> Result<DecisionTrainingStep, DecisionFailure> {
        self.has_unreported_cleanup = true;
        match self.scripted_training_steps.pop_front() {
            Some(scripted_step) => scripted_step,
            None => Err(DecisionFailure::fatal_execution(
                "no training is in progress",
            )),
        }
    }

    fn cancel_training(&mut self) {
        self.scripted_training_steps.clear();
        *self
            .was_training_cancelled
            .lock()
            .expect("the cancellation flag should not be poisoned") = true;
    }

    fn take_post_cleanup_memory_telemetry(&mut self) -> Option<MlxMemoryTelemetry> {
        if !self.has_unreported_cleanup {
            return None;
        }
        self.has_unreported_cleanup = false;
        Some(MlxMemoryTelemetry::new(
            96,
            0,
            512,
            MlxActiveMemoryBreakdown::default(),
        ))
    }
}

/// Two marker states, standing in for what a real head reads.
fn scripted_head_inputs() -> HeadInputs {
    HeadInputs::Laya {
        marker_states: vec![FloatVector(vec![0.25, -1.5]), FloatVector(vec![2.0, 0.5])],
    }
}

fn even_noul_outcome(question_name: &str) -> QuestionOutcome {
    QuestionOutcome {
        question_name: question_name.to_owned(),
        answer: DecisionAnswer::Noul(NoulAnswer {
            probability_true: 0.5,
            answer_confidence: 0.5,
        }),
        option_logits: vec![0.0, 0.0],
        applied_temperature: 1.0,
        input_coverage: QuestionInputCoverage {
            state_token_count: 12,
            state_tokens_read: 12,
            option_token_cap: None,
            distinguishable_option_count: 2,
        },
        head_inputs: None,
        training_record: None,
    }
}

struct ScriptedDecisionFactory {
    decision_engines: Mutex<VecDeque<ScriptedDecisionEngine>>,
}

impl
    ModelFactory<
        ScriptedChatProcessor,
        ScriptedChatEngine,
        ImageGenerationUnavailableEngine,
        EmbeddingUnavailableEngine,
        ScriptedDecisionEngine,
    > for ScriptedDecisionFactory
{
    async fn create(
        &self,
        _model_directory: &str,
        _model_configuration: WorkerModelConfiguration,
    ) -> Result<ScriptedDecisionRuntime, String> {
        let decision_engine = self
            .decision_engines
            .lock()
            .expect("the scripted engine queue should not be poisoned")
            .pop_front()
            .expect("the test should configure every model creation");
        Ok(ModelFactoryRuntime::Decisions(decision_engine))
    }
}

pub(super) struct DecisionsWorkerHarness {
    pub(super) events: DaemonReader,
    pub(super) commands: DaemonWriter,
    worker_task: JoinHandle<Result<(), WorkerRuntimeError>>,
    asked_question_names: Arc<Mutex<Vec<String>>>,
    pub(super) was_training_cancelled: Arc<Mutex<bool>>,
}

/// Starts an idle worker whose factory builds one decision engine that loads
/// as `loaded_model_id`, and asks the worker to swap that model in.
async fn start_worker_and_swap_in_a_decision_model(
    loaded_model_id: &str,
    scripted_outcomes: Vec<Result<(), DecisionFailure>>,
) -> DecisionsWorkerHarness {
    start_worker_with_scripts(loaded_model_id, scripted_outcomes, Vec::new()).await
}

/// Like [`start_worker_and_swap_in_a_decision_model`], for a model whose
/// training takes `scripted_training_steps`, one per advance.
pub(super) async fn start_worker_and_swap_in_a_trainable_model(
    scripted_training_steps: Vec<Result<DecisionTrainingStep, DecisionFailure>>,
) -> DecisionsWorkerHarness {
    start_worker_with_scripts(DECISION_MODEL_ID, Vec::new(), scripted_training_steps).await
}

async fn start_worker_with_scripts(
    loaded_model_id: &str,
    scripted_outcomes: Vec<Result<(), DecisionFailure>>,
    scripted_training_steps: Vec<Result<DecisionTrainingStep, DecisionFailure>>,
) -> DecisionsWorkerHarness {
    let asked_question_names = Arc::new(Mutex::new(Vec::new()));
    let was_training_cancelled = Arc::new(Mutex::new(false));
    let decision_engine = ScriptedDecisionEngine {
        loaded_model_id: loaded_model_id.to_owned(),
        scripted_outcomes: scripted_outcomes.into(),
        asked_question_names: Arc::clone(&asked_question_names),
        has_unreported_cleanup: false,
        scripted_training_steps: scripted_training_steps.into(),
        was_training_cancelled: Arc::clone(&was_training_cancelled),
    };
    let worker: DecisionsWorker = EngineBackedWorker::idle_with_model_factory(
        ScriptedDecisionFactory {
            decision_engines: Mutex::new(VecDeque::from([decision_engine])),
        },
        8_000,
    );
    let (daemon_transport, worker_transport) = duplex(MAX_IPC_FRAME_BYTES * 2);
    let (daemon_read, daemon_write) = split(daemon_transport);
    let (worker_read, worker_write) = split(worker_transport);
    let worker_task = tokio::spawn(async move { worker.run(worker_read, worker_write).await });
    let mut harness = DecisionsWorkerHarness {
        events: ProtocolReader::new(daemon_read),
        commands: ProtocolWriter::new(daemon_write),
        worker_task,
        asked_question_names,
        was_training_cancelled,
    };
    assert!(matches!(
        next_event(&mut harness.events).await,
        WorkerEvent::Idle { .. }
    ));
    harness
        .commands
        .send_command(&WorkerCommand::SwapModel {
            model_directory: "/models/decision".to_owned(),
            model_configuration: WorkerModelConfiguration::Decisions(
                WorkerDecisionModelConfiguration {
                    model_id: DECISION_MODEL_ID.to_owned(),
                    model_family: WorkerDecisionModelFamily::Laya,
                    artifact_revision: "scripted-revision".to_owned(),
                    maximum_tokens_per_question: MAXIMUM_TOKENS_PER_QUESTION,
                },
            ),
        })
        .await
        .expect("the swap command should reach the worker");
    harness
}

/// The next event that is not a periodic memory sample.
pub(super) async fn next_event(events: &mut DaemonReader) -> WorkerEvent {
    loop {
        let worker_event = super::next_event_skipping_load_progress(events).await;
        if !matches!(worker_event, WorkerEvent::MlxMemorySample { .. }) {
            return worker_event;
        }
    }
}

fn decisions_command(request_number: u64, questions_json: &str) -> WorkerCommand {
    WorkerCommand::GenerateDecisions(DecisionsCommand {
        request_id: RequestId::new(request_number),
        model: DECISION_MODEL_ID.to_owned(),
        state: OrderedJsonValue::Text(HAMLET_LINE.to_owned()),
        questions: serde_json::from_str(questions_json).expect("the test questions should be JSON"),
    })
}

pub(super) async fn close(harness: DecisionsWorkerHarness) {
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

#[tokio::test]
async fn should_swap_in_a_decision_model_and_advertise_its_token_limit() {
    let mut harness = start_worker_and_swap_in_a_decision_model(DECISION_MODEL_ID, vec![]).await;

    let swapped_event = next_event(&mut harness.events).await;

    let WorkerEvent::ModelSwapped {
        model_id,
        capabilities,
        ..
    } = swapped_event
    else {
        panic!("the worker should announce the swapped decision model: {swapped_event:?}");
    };
    assert_eq!(model_id, DECISION_MODEL_ID);
    assert_eq!(
        capabilities
            .decisions
            .expect("a decision model advertises its decisions surface")
            .maximum_tokens_per_question,
        MAXIMUM_TOKENS_PER_QUESTION
    );
    assert!(capabilities.chat.is_none() && capabilities.embeddings.is_none());
    close(harness).await;
}

#[tokio::test]
async fn should_refuse_a_decision_model_that_loads_under_another_identity() {
    let mut harness =
        start_worker_and_swap_in_a_decision_model("some-other-decision-model", vec![]).await;

    let swap_event = next_event(&mut harness.events).await;

    assert!(
        matches!(swap_event, WorkerEvent::ModelSwapFailed { .. }),
        "a model that loads as a different identity must not be served: {swap_event:?}"
    );
    close(harness).await;
}

#[tokio::test]
async fn should_answer_questions_in_request_order_then_finalize_with_memory() {
    let mut harness =
        start_worker_and_swap_in_a_decision_model(DECISION_MODEL_ID, vec![Ok(())]).await;
    assert!(matches!(
        next_event(&mut harness.events).await,
        WorkerEvent::ModelSwapped { .. }
    ));

    harness
        .commands
        .send_command(&decisions_command(
            41,
            r#"{"tragic": {"type": "noul", "instructions": "Is the speaker in despair?"},
                "asking": {"type": "noul", "instructions": "Is a question being asked?"}}"#,
        ))
        .await
        .expect("the decisions command should reach the worker");

    let completed_event = next_event(&mut harness.events).await;
    let WorkerEvent::DecisionsCompleted {
        request_id,
        question_outcomes,
        input_token_count,
        ..
    } = completed_event
    else {
        panic!("the worker should complete the request: {completed_event:?}");
    };
    assert_eq!(request_id, RequestId::new(41));
    assert_eq!(input_token_count, 24);
    let answered_question_names: Vec<&str> = question_outcomes
        .iter()
        .map(|question_outcome| question_outcome.question_name.as_str())
        .collect();
    assert_eq!(answered_question_names, ["tragic", "asking"]);
    assert!(
        matches!(next_event(&mut harness.events).await, WorkerEvent::DecisionsFinalized { request_id, mlx_memory_snapshot: Some(snapshot), .. } if request_id == RequestId::new(41) && snapshot.source == MlxMemorySnapshotSource::Finalized)
    );
    assert_eq!(
        *harness
            .asked_question_names
            .lock()
            .expect("the asked-question log should not be poisoned"),
        ["tragic", "asking"]
    );
    close(harness).await;
}

#[tokio::test]
async fn should_never_send_head_inputs_back_with_an_answer() {
    let mut harness =
        start_worker_and_swap_in_a_decision_model(DECISION_MODEL_ID, vec![Ok(())]).await;
    assert!(matches!(
        next_event(&mut harness.events).await,
        WorkerEvent::ModelSwapped { .. }
    ));
    let questions_json =
        r#"{"tragic": {"type": "noul", "instructions": "Is the speaker in despair?"}}"#;
    harness
        .commands
        .send_command(&decisions_command(51, questions_json))
        .await
        .expect("the decisions command should reach the worker");
    let completed_event = next_event(&mut harness.events).await;
    let WorkerEvent::DecisionsCompleted {
        question_outcomes, ..
    } = completed_event
    else {
        panic!("the worker should complete the request: {completed_event:?}");
    };
    // The scripted engine returns head inputs whenever it is asked for
    // them; only training reads them, inside the worker, so an answer
    // request never asks and its outcomes stay small.
    assert_eq!(question_outcomes[0].head_inputs, None);
    assert!(matches!(
        next_event(&mut harness.events).await,
        WorkerEvent::DecisionsFinalized { .. }
    ));
    close(harness).await;
}

#[tokio::test]
async fn should_fail_malformed_questions_without_asking_the_model_and_stay_usable() {
    let mut harness =
        start_worker_and_swap_in_a_decision_model(DECISION_MODEL_ID, vec![Ok(())]).await;
    assert!(matches!(
        next_event(&mut harness.events).await,
        WorkerEvent::ModelSwapped { .. }
    ));

    harness
        .commands
        .send_command(&decisions_command(50, "{}"))
        .await
        .expect("the malformed command should reach the worker");
    assert!(
        matches!(next_event(&mut harness.events).await, WorkerEvent::DecisionsFailed { request_id, reason: DecisionFailure::InvalidRequest { .. } } if request_id == RequestId::new(50))
    );
    assert!(
        matches!(next_event(&mut harness.events).await, WorkerEvent::DecisionsFinalized { request_id, mlx_memory_snapshot: None, .. } if request_id == RequestId::new(50)),
        "a request the model never saw has no cleanup observation to report"
    );
    assert!(
        harness
            .asked_question_names
            .lock()
            .expect("the asked-question log should not be poisoned")
            .is_empty()
    );

    harness
        .commands
        .send_command(&decisions_command(
            51,
            r#"{"asking": {"type": "noul", "instructions": "Is a question being asked?"}}"#,
        ))
        .await
        .expect("the next command should reach the worker");
    assert!(
        matches!(next_event(&mut harness.events).await, WorkerEvent::DecisionsCompleted { request_id, .. } if request_id == RequestId::new(51)),
        "a refused request must leave the model ready for the next one"
    );
    close(harness).await;
}

#[tokio::test]
async fn should_report_an_engine_failure_as_the_requests_outcome_and_finalize() {
    let mut harness = start_worker_and_swap_in_a_decision_model(
        DECISION_MODEL_ID,
        vec![Err(DecisionFailure::fatal_execution(
            "the scripted forward pass failed",
        ))],
    )
    .await;
    assert!(matches!(
        next_event(&mut harness.events).await,
        WorkerEvent::ModelSwapped { .. }
    ));

    harness
        .commands
        .send_command(&decisions_command(
            60,
            r#"{"asking": {"type": "noul", "instructions": "Is a question being asked?"}}"#,
        ))
        .await
        .expect("the decisions command should reach the worker");

    assert!(
        matches!(next_event(&mut harness.events).await, WorkerEvent::DecisionsFailed { request_id, reason: DecisionFailure::FatalExecution { .. } } if request_id == RequestId::new(60))
    );
    assert!(
        matches!(next_event(&mut harness.events).await, WorkerEvent::DecisionsFinalized { request_id, .. } if request_id == RequestId::new(60))
    );
    close(harness).await;
}
