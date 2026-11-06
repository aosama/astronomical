//! Neural Engine programs prepared while the worker is idle, as the daemon sees it.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use astronomical_ipc_protocol::{
    MAX_IPC_FRAME_BYTES, NeuralEnginePrefillStatus, ProtocolReader, ProtocolWriter, RequestId,
    WorkerCommand, WorkerEvent,
};
use tokio::io::{duplex, split};
use tokio::time::timeout;

use super::scripted_chat_test_doubles::ScriptedChatProcessor;
use super::support::{ScriptedInferenceRequest, chat_command, next_raw_event};
use crate::{
    EngineBackedWorker, EngineGenerationStart, EngineLoadResult, GeneratedToken,
    GenerationFinalization, InferenceEngine, InferenceEngineError,
};

const STEP_LIMIT: Duration = Duration::from_secs(10);

fn preparing(completed_program_count: u32) -> NeuralEnginePrefillStatus {
    NeuralEnginePrefillStatus::Preparing {
        completed_program_count,
        total_program_count: 2,
        elapsed_ms: 0,
        estimated_remaining_ms: None,
        is_waiting_to_measure: false,
    }
}

fn active() -> NeuralEnginePrefillStatus {
    NeuralEnginePrefillStatus::Active {
        layer_program_count: 2,
        neural_engine_tokens_per_chunk: 64,
        chunk_token_count: 128,
        measured_memory_bytes: 1,
        measured_speed_percent: None,
    }
}

/// Answers each preparation step from a script, then with nothing new.
struct PreparingEngine {
    status_at_load: NeuralEnginePrefillStatus,
    scripted_steps: VecDeque<Option<NeuralEnginePrefillStatus>>,
    step_count: Arc<AtomicUsize>,
    owner_thread_has_stopped: bool,
}

impl InferenceEngine for PreparingEngine {
    type Request = ScriptedInferenceRequest;

    async fn load(&mut self) -> Result<EngineLoadResult, InferenceEngineError> {
        Ok(EngineLoadResult::new().with_neural_engine_prefill_status(self.status_at_load.clone()))
    }

    async fn advance_neural_engine_preparation(
        &mut self,
    ) -> Result<Option<NeuralEnginePrefillStatus>, InferenceEngineError> {
        self.step_count.fetch_add(1, Ordering::SeqCst);
        if self.owner_thread_has_stopped {
            return Err(InferenceEngineError::Fatal {
                reason: "the owner thread stopped".to_owned(),
            });
        }
        Ok(self.scripted_steps.pop_front().flatten())
    }

    async fn start_generation(
        &mut self,
        _generation_request: ScriptedInferenceRequest,
    ) -> Result<EngineGenerationStart, InferenceEngineError> {
        Ok(EngineGenerationStart::new(0))
    }

    async fn decode_next_token(
        &mut self,
        _request_id: RequestId,
    ) -> Result<GeneratedToken, InferenceEngineError> {
        Ok(GeneratedToken::EndOfSequence)
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

/// Everything the worker says about its programs, up to and including `last_status`.
async fn prefill_statuses_through<ReadTransport>(
    daemon_reader: &mut ProtocolReader<ReadTransport>,
    last_status: &NeuralEnginePrefillStatus,
) -> Vec<NeuralEnginePrefillStatus>
where
    ReadTransport: tokio::io::AsyncRead + Unpin,
{
    let mut prefill_statuses = Vec::new();
    loop {
        let worker_event = timeout(STEP_LIMIT, next_raw_event(daemon_reader))
            .await
            .expect("the worker should keep reporting");
        if let WorkerEvent::NeuralEnginePrefillChanged {
            neural_engine_prefill,
        } = worker_event
        {
            let is_last = &neural_engine_prefill == last_status;
            prefill_statuses.push(neural_engine_prefill);
            if is_last {
                return prefill_statuses;
            }
        }
    }
}

fn engine_with_steps(
    status_at_load: NeuralEnginePrefillStatus,
    scripted_steps: Vec<Option<NeuralEnginePrefillStatus>>,
) -> (PreparingEngine, Arc<AtomicUsize>) {
    let step_count = Arc::new(AtomicUsize::new(0));
    let engine = PreparingEngine {
        status_at_load,
        scripted_steps: scripted_steps.into(),
        step_count: Arc::clone(&step_count),
        owner_thread_has_stopped: false,
    };
    (engine, step_count)
}

#[tokio::test]
async fn should_report_each_step_in_order_until_the_programs_are_engaged() {
    let (engine, _step_count) = engine_with_steps(
        preparing(0),
        vec![None, Some(preparing(1)), None, Some(active())],
    );
    let worker = EngineBackedWorker::new(ScriptedChatProcessor::with_prompt_token_count(1), engine);
    let (daemon_transport, worker_transport) = duplex(MAX_IPC_FRAME_BYTES * 2);
    let (daemon_read, _daemon_write) = split(daemon_transport);
    let (worker_read, worker_write) = split(worker_transport);
    let worker_task = tokio::spawn(worker.run(worker_read, worker_write));
    let mut daemon_reader = ProtocolReader::new(daemon_read);

    let prefill_statuses = prefill_statuses_through(&mut daemon_reader, &active()).await;

    assert_eq!(prefill_statuses, vec![preparing(0), preparing(1), active()]);
    worker_task.abort();
}

#[tokio::test]
async fn should_stop_stepping_once_the_programs_are_settled() {
    let (engine, step_count) = engine_with_steps(preparing(0), vec![Some(active())]);
    let worker = EngineBackedWorker::new(ScriptedChatProcessor::with_prompt_token_count(1), engine);
    let (daemon_transport, worker_transport) = duplex(MAX_IPC_FRAME_BYTES * 2);
    let (daemon_read, _daemon_write) = split(daemon_transport);
    let (worker_read, worker_write) = split(worker_transport);
    let worker_task = tokio::spawn(worker.run(worker_read, worker_write));
    let mut daemon_reader = ProtocolReader::new(daemon_read);
    prefill_statuses_through(&mut daemon_reader, &active()).await;

    tokio::time::sleep(Duration::from_millis(300)).await;

    assert_eq!(step_count.load(Ordering::SeqCst), 1);
    worker_task.abort();
}

#[tokio::test]
async fn should_never_step_a_model_with_nothing_to_prepare() {
    let (engine, step_count) = engine_with_steps(
        NeuralEnginePrefillStatus::Unavailable {
            reason: "only dense models are supported".to_owned(),
        },
        vec![Some(active())],
    );
    let worker = EngineBackedWorker::new(ScriptedChatProcessor::with_prompt_token_count(1), engine);
    let (daemon_transport, worker_transport) = duplex(MAX_IPC_FRAME_BYTES * 2);
    let (daemon_read, _daemon_write) = split(daemon_transport);
    let (worker_read, worker_write) = split(worker_transport);
    let worker_task = tokio::spawn(worker.run(worker_read, worker_write));
    let mut daemon_reader = ProtocolReader::new(daemon_read);
    prefill_statuses_through(
        &mut daemon_reader,
        &NeuralEnginePrefillStatus::Unavailable {
            reason: "only dense models are supported".to_owned(),
        },
    )
    .await;

    tokio::time::sleep(Duration::from_millis(300)).await;

    assert_eq!(step_count.load(Ordering::SeqCst), 0);
    worker_task.abort();
}

#[tokio::test]
async fn should_end_cleanly_when_the_daemon_leaves_during_a_preparation() {
    let (engine, _step_count) = engine_with_steps(preparing(0), Vec::new());
    let worker = EngineBackedWorker::new(ScriptedChatProcessor::with_prompt_token_count(1), engine);
    let (daemon_transport, worker_transport) = duplex(MAX_IPC_FRAME_BYTES * 2);
    let (daemon_read, daemon_write) = split(daemon_transport);
    let (worker_read, worker_write) = split(worker_transport);
    let worker_task = tokio::spawn(worker.run(worker_read, worker_write));
    let mut daemon_reader = ProtocolReader::new(daemon_read);
    prefill_statuses_through(&mut daemon_reader, &preparing(0)).await;

    drop(daemon_write);
    drop(daemon_reader);

    timeout(STEP_LIMIT, worker_task)
        .await
        .expect("the worker should notice the daemon left")
        .expect("the worker task should not panic")
        .expect("the worker should end without an error");
}

#[tokio::test]
async fn should_end_the_count_when_the_owner_thread_has_stopped() {
    let (mut engine, step_count) = engine_with_steps(preparing(0), Vec::new());
    engine.owner_thread_has_stopped = true;
    let worker = EngineBackedWorker::new(ScriptedChatProcessor::with_prompt_token_count(1), engine);
    let (daemon_transport, worker_transport) = duplex(MAX_IPC_FRAME_BYTES * 2);
    let (daemon_read, _daemon_write) = split(daemon_transport);
    let (worker_read, worker_write) = split(worker_transport);
    let worker_task = tokio::spawn(worker.run(worker_read, worker_write));
    let mut daemon_reader = ProtocolReader::new(daemon_read);
    let stopped_status = NeuralEnginePrefillStatus::Unavailable {
        reason: "the Neural Engine preparation stopped".to_owned(),
    };

    let prefill_statuses = prefill_statuses_through(&mut daemon_reader, &stopped_status).await;
    tokio::time::sleep(Duration::from_millis(300)).await;

    assert_eq!(prefill_statuses, vec![preparing(0), stopped_status]);
    assert_eq!(step_count.load(Ordering::SeqCst), 1);
    worker_task.abort();
}

#[tokio::test]
async fn should_report_what_a_request_settled_about_running_programs() {
    let slower_status = NeuralEnginePrefillStatus::Unavailable {
        reason: "the Neural Engine ran this model at 0.91 times the speed of the GPU alone"
            .to_owned(),
    };
    let (engine, step_count) = engine_with_steps(active(), vec![Some(slower_status.clone())]);
    let worker = EngineBackedWorker::new(ScriptedChatProcessor::with_prompt_token_count(1), engine);
    let (daemon_transport, worker_transport) = duplex(MAX_IPC_FRAME_BYTES * 2);
    let (daemon_read, daemon_write) = split(daemon_transport);
    let (worker_read, worker_write) = split(worker_transport);
    let worker_task = tokio::spawn(worker.run(worker_read, worker_write));
    let mut daemon_reader = ProtocolReader::new(daemon_read);
    let mut daemon_writer = ProtocolWriter::new(daemon_write);
    prefill_statuses_through(&mut daemon_reader, &active()).await;
    assert_eq!(step_count.load(Ordering::SeqCst), 0);

    daemon_writer
        .send_command(&WorkerCommand::Generate(chat_command(811, 7)))
        .await
        .expect("the request is sent");
    let prefill_statuses = prefill_statuses_through(&mut daemon_reader, &slower_status).await;
    tokio::time::sleep(Duration::from_millis(300)).await;

    assert_eq!(prefill_statuses, vec![slower_status]);
    assert_eq!(step_count.load(Ordering::SeqCst), 1);
    worker_task.abort();
}
