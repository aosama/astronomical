//! The live memory-limit journeys every non-chat kind shares (issue #2681):
//! a change is applied and the model keeps serving, a change below the floor
//! is refused and the model keeps serving, and an engine that cannot move its
//! ceiling answers with a clean rejection instead of leaving the worker.

use ipc_protocol::{WorkerCommand, WorkerEvent};
use memory_policy::live_memory_limit::{LiveMemoryLimitRefusal, live_ceiling_refusal};

use super::daemon_side::WorkerUnderTest;
use crate::MlxMemoryLimitAdjustment;

/// What the scripted model holds; a ceiling below it is refused.
pub(super) const SCRIPTED_FLOOR_BYTES: u64 = 1_000;

/// A scripted engine's answer to a live ceiling change: `None` is an engine
/// that keeps the default and cannot move its ceiling.
#[derive(Clone, Copy)]
pub(super) struct ScriptedLiveCeiling {
    pub(super) floor_bytes: Option<u64>,
}

impl ScriptedLiveCeiling {
    pub(super) fn with_floor(floor_bytes: Option<u64>) -> Self {
        Self { floor_bytes }
    }

    pub(super) fn answer(
        &self,
        requested_mlx_memory_ceiling_bytes: u64,
    ) -> Result<MlxMemoryLimitAdjustment, LiveMemoryLimitRefusal> {
        let Some(floor_bytes) = self.floor_bytes else {
            return Err(LiveMemoryLimitRefusal::Unsupported);
        };
        if let Some(refusal) = live_ceiling_refusal(requested_mlx_memory_ceiling_bytes, floor_bytes)
        {
            return Err(refusal);
        }
        Ok(MlxMemoryLimitAdjustment::resident(
            requested_mlx_memory_ceiling_bytes,
            0,
            floor_bytes,
            None,
        ))
    }
}

pub(super) fn memory_limit_command(effective_mlx_memory_ceiling_bytes: u64) -> WorkerCommand {
    WorkerCommand::UpdateMlxMemoryLimit {
        effective_mlx_memory_ceiling_bytes,
        configuration_generation: "scripted-generation".to_owned(),
    }
}

/// One more request for the loaded model, and how to recognise its answer.
/// A function, not a closure, because the journeys only need to match an
/// event's kind.
pub(super) struct OneMoreRequest {
    pub(super) worker_command: WorkerCommand,
    pub(super) is_its_completion: fn(&WorkerEvent) -> bool,
}

/// Sends the request and reads its events until it completes. A handful of
/// events is all any scripted request sends, so a longer run is a failure.
async fn serves_one_more_request(
    worker_under_test: &mut WorkerUnderTest,
    one_more_request: OneMoreRequest,
) {
    worker_under_test
        .send(one_more_request.worker_command)
        .await;
    for _ in 0..8 {
        let worker_event = worker_under_test.next_event().await;
        if (one_more_request.is_its_completion)(&worker_event) {
            return;
        }
    }
    panic!("the loaded model did not complete one more request");
}

pub(super) async fn assert_a_change_applies_and_the_model_keeps_serving(
    mut worker_under_test: WorkerUnderTest,
    one_more_request: OneMoreRequest,
) {
    worker_under_test.send(memory_limit_command(4_000)).await;

    assert!(matches!(
        worker_under_test.next_event().await,
        WorkerEvent::MlxMemoryLimitChanged {
            effective_mlx_memory_ceiling_bytes: 4_000,
            ..
        }
    ));
    serves_one_more_request(&mut worker_under_test, one_more_request).await;
    worker_under_test.close().await;
}

/// One byte under the floor is refused with the floor named; exactly the
/// floor is accepted; the model serves throughout.
pub(super) async fn assert_a_change_below_the_floor_is_refused_and_the_model_keeps_serving(
    mut worker_under_test: WorkerUnderTest,
    one_more_request: OneMoreRequest,
) {
    worker_under_test
        .send(memory_limit_command(SCRIPTED_FLOOR_BYTES - 1))
        .await;

    assert!(matches!(
        worker_under_test.next_event().await,
        WorkerEvent::MlxMemoryLimitRejected {
            requested_mlx_memory_ceiling_bytes,
            minimum_mlx_memory_ceiling_bytes: SCRIPTED_FLOOR_BYTES,
            ..
        } if requested_mlx_memory_ceiling_bytes == SCRIPTED_FLOOR_BYTES - 1
    ));
    worker_under_test
        .send(memory_limit_command(SCRIPTED_FLOOR_BYTES))
        .await;
    assert!(matches!(
        worker_under_test.next_event().await,
        WorkerEvent::MlxMemoryLimitChanged {
            effective_mlx_memory_ceiling_bytes: SCRIPTED_FLOOR_BYTES,
            ..
        }
    ));
    serves_one_more_request(&mut worker_under_test, one_more_request).await;
    worker_under_test.close().await;
}

pub(super) async fn assert_an_engine_that_cannot_move_its_ceiling_is_rejected_cleanly(
    mut worker_under_test: WorkerUnderTest,
    one_more_request: OneMoreRequest,
) {
    worker_under_test.send(memory_limit_command(4_000)).await;

    assert!(matches!(
        worker_under_test.next_event().await,
        WorkerEvent::MlxMemoryLimitRejected {
            requested_mlx_memory_ceiling_bytes: 4_000,
            ..
        }
    ));
    serves_one_more_request(&mut worker_under_test, one_more_request).await;
    worker_under_test.close().await;
}
