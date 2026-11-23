use std::sync::Arc;

use ipc_protocol::RequestId;
use tokio::sync::{Semaphore, mpsc};
use tokio::time::Instant;

use super::AdmittedRequest;
use crate::worker::request_admission::SINGLE_SHOT_EXECUTION_TIMEOUT;

fn admitted_request_at(execution_started_at: Instant) -> AdmittedRequest<()> {
    let (result_sender, _result_receiver) = mpsc::channel(1);
    AdmittedRequest {
        active_generation_permit: Arc::new(Semaphore::new(1))
            .try_acquire_owned()
            .expect("the test semaphore permit should be acquirable"),
        request_id: RequestId::new(7),
        execution_started_at,
        result_sender,
    }
}

// Positive: a forward-pass request is bounded from the moment it was
// dispatched, not from when it was queued.
#[test]
fn should_bound_a_forward_pass_from_its_dispatch() {
    let execution_started_at = Instant::now();

    let admitted_request = admitted_request_at(execution_started_at);

    assert_eq!(
        admitted_request.single_shot_execution_deadline(),
        execution_started_at + SINGLE_SHOT_EXECUTION_TIMEOUT
    );
}

// Edge: the deadline is fixed at dispatch, so reading it later does not
// move it.
#[test]
fn should_keep_the_same_deadline_when_read_again() {
    let admitted_request = admitted_request_at(Instant::now());

    let first_deadline = admitted_request.single_shot_execution_deadline();
    let second_deadline = admitted_request.single_shot_execution_deadline();

    assert_eq!(first_deadline, second_deadline);
}
