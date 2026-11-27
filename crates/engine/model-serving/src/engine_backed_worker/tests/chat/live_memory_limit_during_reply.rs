//! A memory-limit change sent while a reply is being generated (issue #2681):
//! the engine answers it between two tokens and the reply goes on.

use super::chat_request_exchange::send_chat_request;
use super::ipc_worker_transport::open_ipc_worker_transport;
use super::scripted_chat_test_doubles::{
    REFUSED_CEILING_REASON, ScriptedChatEngine, ScriptedChatProcessor,
};
use super::support::{close_worker_transport, next_event, ready_event};
use crate::EngineBackedWorker;
use ipc_protocol::{WorkerCommand, WorkerEvent};
use std::sync::Arc;

const MACHINE_MLX_MEMORY_CEILING_BYTES: u64 = 10_000;
const REQUESTED_REPLY_NUMBER: u64 = 931;

/// What the daemon saw of one reply, with the limit answers kept apart so the
/// reply's own events can be compared with a reply that had no change.
struct ObservedReply {
    reply_events: Vec<WorkerEvent>,
    limit_answers: Vec<WorkerEvent>,
}

/// Runs one reply on a worker whose engine refuses ceilings below
/// `refused_ceiling_floor_bytes` (when given), sending `requested_ceiling_bytes`
/// right behind the request, and returns what the daemon saw plus the
/// ceilings the engine adopted.
async fn observe_reply_with_limit_change(
    requested_ceiling_bytes: Option<u64>,
    refused_ceiling_floor_bytes: Option<u64>,
) -> (ObservedReply, Vec<u64>) {
    let mut scripted_engine = ScriptedChatEngine::new();
    if let Some(floor_bytes) = refused_ceiling_floor_bytes {
        scripted_engine = scripted_engine.with_refused_ceilings_below(floor_bytes);
    }
    let adopted_ceilings = Arc::clone(&scripted_engine.adopted_ceilings);
    let mut engine_worker = EngineBackedWorker::new(ScriptedChatProcessor::new(), scripted_engine);
    engine_worker.machine_mlx_memory_ceiling_bytes = MACHINE_MLX_MEMORY_CEILING_BYTES;
    let (mut daemon_reader, mut daemon_writer, worker_reader_transport, worker_writer_transport) =
        open_ipc_worker_transport();
    let worker_task =
        tokio::spawn(engine_worker.run(worker_reader_transport, worker_writer_transport));
    assert_eq!(next_event(&mut daemon_reader).await, ready_event());

    send_chat_request(&mut daemon_writer, REQUESTED_REPLY_NUMBER, 7).await;
    if let Some(requested_ceiling_bytes) = requested_ceiling_bytes {
        daemon_writer
            .send_command(&WorkerCommand::UpdateMlxMemoryLimit {
                effective_mlx_memory_ceiling_bytes: requested_ceiling_bytes,
                configuration_generation: "generation-during-reply".to_owned(),
            })
            .await
            .expect("the limit change should reach the worker during the reply");
    }
    let mut observed_reply = ObservedReply {
        reply_events: Vec::new(),
        limit_answers: Vec::new(),
    };
    loop {
        let worker_event = next_event(&mut daemon_reader).await;
        match worker_event {
            WorkerEvent::MlxMemoryLimitChanged { .. }
            | WorkerEvent::MlxMemoryLimitRejected { .. } => {
                observed_reply.limit_answers.push(worker_event);
            }
            WorkerEvent::Completed { .. } => {
                observed_reply.reply_events.push(worker_event);
                break;
            }
            other_event => observed_reply.reply_events.push(other_event),
        }
    }
    close_worker_transport(daemon_writer, worker_task).await;
    let adopted = adopted_ceilings
        .lock()
        .expect("the adopted-ceiling record should not be poisoned")
        .clone();
    (observed_reply, adopted)
}

#[tokio::test]
async fn should_apply_a_limit_change_during_a_reply_and_finish_the_reply_in_order() {
    let (plain_reply, _) = observe_reply_with_limit_change(None, None).await;

    let (changed_reply, adopted_ceilings) =
        observe_reply_with_limit_change(Some(4_000), None).await;

    assert_eq!(adopted_ceilings, vec![4_000]);
    assert_eq!(changed_reply.limit_answers.len(), 1);
    assert!(matches!(
        changed_reply.limit_answers[0],
        WorkerEvent::MlxMemoryLimitChanged {
            effective_mlx_memory_ceiling_bytes: 4_000,
            ..
        }
    ));
    assert_eq!(
        changed_reply.reply_events, plain_reply.reply_events,
        "the reply must read the same with or without the change"
    );
}

#[tokio::test]
async fn should_refuse_a_limit_change_the_reply_cannot_honour_and_finish_the_reply() {
    let (plain_reply, _) = observe_reply_with_limit_change(None, None).await;

    let (refused_reply, adopted_ceilings) =
        observe_reply_with_limit_change(Some(2_000), Some(3_000)).await;

    assert!(adopted_ceilings.is_empty());
    assert_eq!(refused_reply.limit_answers.len(), 1);
    let WorkerEvent::MlxMemoryLimitRejected {
        requested_mlx_memory_ceiling_bytes,
        minimum_mlx_memory_ceiling_bytes,
        reason,
        ..
    } = &refused_reply.limit_answers[0]
    else {
        panic!(
            "the change should be refused: {:?}",
            refused_reply.limit_answers
        );
    };
    assert_eq!(*requested_mlx_memory_ceiling_bytes, 2_000);
    assert_eq!(*minimum_mlx_memory_ceiling_bytes, 3_000);
    assert_eq!(reason, REFUSED_CEILING_REASON);
    assert_eq!(refused_reply.reply_events, plain_reply.reply_events);
}

#[tokio::test]
async fn should_accept_a_limit_change_that_equals_the_floor_the_reply_needs() {
    let (accepted_reply, adopted_ceilings) =
        observe_reply_with_limit_change(Some(3_000), Some(3_000)).await;

    assert_eq!(adopted_ceilings, vec![3_000]);
    assert!(matches!(
        accepted_reply.limit_answers[0],
        WorkerEvent::MlxMemoryLimitChanged { .. }
    ));
}
