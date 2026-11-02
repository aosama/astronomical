use super::scripted_chat_test_doubles::ScriptedChatProcessor;
use super::support::{chat_command, close_worker_transport, next_event};
use super::tracking_chat_engine_test_doubles::TrackingChatEngine;
use crate::EngineBackedWorker;
use astronomical_ipc_protocol::{
    ChatGenerationCompletionReason, ChatGenerationFailureReason, ChatToolChoice,
    MAX_IPC_FRAME_BYTES, ProtocolReader, ProtocolWriter, RequestId, WorkerCommand, WorkerEvent,
};
use tokio::io::{DuplexStream, ReadHalf, WriteHalf, duplex, split};
use tokio::task::JoinHandle;

type DaemonSide = (
    ProtocolReader<ReadHalf<DuplexStream>>,
    ProtocolWriter<WriteHalf<DuplexStream>>,
    JoinHandle<Result<(), crate::WorkerRuntimeError>>,
);

/// A worker whose processor prepares every prompt to `prompt_token_count`
/// tokens and whose engine refuses a second start while one is active.
async fn worker_with_prepared_prompt_length(prompt_token_count: usize) -> DaemonSide {
    let engine_worker = EngineBackedWorker::new(
        ScriptedChatProcessor::with_prompt_token_count(prompt_token_count),
        TrackingChatEngine::new(),
    );
    let (daemon_transport, worker_transport) = duplex(MAX_IPC_FRAME_BYTES * 2);
    let (daemon_reader_transport, daemon_writer_transport) = split(daemon_transport);
    let (worker_reader_transport, worker_writer_transport) = split(worker_transport);
    let mut daemon_reader = ProtocolReader::new(daemon_reader_transport);
    let worker_task = tokio::spawn(async move {
        engine_worker
            .run(worker_reader_transport, worker_writer_transport)
            .await
    });
    assert!(matches!(
        next_event(&mut daemon_reader).await,
        WorkerEvent::Ready { .. }
    ));
    (
        daemon_reader,
        ProtocolWriter::new(daemon_writer_transport),
        worker_task,
    )
}

#[tokio::test]
async fn should_answer_a_count_with_the_prepared_prompt_length_and_no_generation() {
    let (mut daemon_reader, mut daemon_writer, worker_task) =
        worker_with_prepared_prompt_length(4_321).await;

    daemon_writer
        .send_command(&WorkerCommand::CountChatTokens(chat_command(731, 3)))
        .await
        .expect("the worker should receive the count");

    assert_eq!(
        next_event(&mut daemon_reader).await,
        WorkerEvent::Completed {
            request_id: RequestId::new(731),
            prompt_token_count: 4_321,
            generated_token_count: 0,
            reasoning_token_count: 0,
            cached_token_count: 0,
            persistent_prompt_cache_diagnostics: None,
            reason: ChatGenerationCompletionReason::EndOfSequence,
        }
    );
    close_worker_transport(daemon_writer, worker_task).await;
}

#[tokio::test]
async fn should_leave_the_engine_idle_so_a_following_generation_starts() {
    let (mut daemon_reader, mut daemon_writer, worker_task) =
        worker_with_prepared_prompt_length(12).await;
    daemon_writer
        .send_command(&WorkerCommand::CountChatTokens(chat_command(732, 3)))
        .await
        .expect("the worker should receive the count");
    assert!(matches!(
        next_event(&mut daemon_reader).await,
        WorkerEvent::Completed { .. }
    ));

    daemon_writer
        .send_command(&WorkerCommand::Generate(chat_command(733, 3)))
        .await
        .expect("the worker should receive the generation");

    // The tracking engine refuses a second start while one is active, so a
    // count that had started the engine would surface here as EngineBusy.
    let first_generation_event = next_event(&mut daemon_reader).await;
    assert!(
        !matches!(
            first_generation_event,
            WorkerEvent::Failed {
                reason: ChatGenerationFailureReason::EngineBusy,
                ..
            }
        ),
        "the count left the engine busy: {first_generation_event:?}"
    );
    close_worker_transport(daemon_writer, worker_task).await;
}

#[tokio::test]
async fn should_reject_a_count_whose_command_fails_validation() {
    let (mut daemon_reader, mut daemon_writer, worker_task) =
        worker_with_prepared_prompt_length(12).await;
    let mut invalid_count_command = chat_command(734, 3);
    invalid_count_command.tool_choice = ChatToolChoice::Required;

    daemon_writer
        .send_command(&WorkerCommand::CountChatTokens(invalid_count_command))
        .await
        .expect("the worker should receive the count");

    assert!(matches!(
        next_event(&mut daemon_reader).await,
        WorkerEvent::Failed {
            request_id,
            reason: ChatGenerationFailureReason::InvalidRequest { .. },
        } if request_id == RequestId::new(734)
    ));
    close_worker_transport(daemon_writer, worker_task).await;
}
