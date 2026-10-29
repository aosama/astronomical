//! Unit contracts for the pure decision units of the worker command arms:
//! the generation readiness guard and the shared worker-replacement rejection.

use crate::worker::command_arms::{reject_unavailable_worker, reject_worker_replacement_request};
use crate::worker::control_error::WorkerControlError;

#[test]
fn should_admit_a_generation_when_the_worker_is_ready_and_running() {
    assert!(
        !reject_unavailable_worker(true, true),
        "a ready worker with a live process must admit generation commands"
    );
}

#[test]
fn should_reject_a_generation_when_the_worker_is_not_ready() {
    assert!(
        reject_unavailable_worker(false, true),
        "a worker still loading its model must reject generation commands"
    );
}

#[test]
fn should_reject_a_generation_when_the_worker_process_is_not_running() {
    assert!(
        reject_unavailable_worker(true, false),
        "a worker without a live process must reject generation commands"
    );
}

#[test]
fn should_reject_a_generation_when_the_worker_is_neither_ready_nor_running() {
    assert!(
        reject_unavailable_worker(false, false),
        "the boundary combination must still reject"
    );
}

#[test]
fn should_admit_an_idle_ready_worker_replacement() {
    let rejection = reject_worker_replacement_request(false, true);
    assert!(
        rejection.is_none(),
        "an idle, ready worker must accept a replacement request"
    );
}

#[test]
fn should_reject_a_worker_replacement_while_a_generation_is_busy() {
    let rejection = reject_worker_replacement_request(true, true);
    assert!(
        matches!(rejection, Some(WorkerControlError::GenerationBusy)),
        "an active generation must reject the replacement with GenerationBusy"
    );
}

#[test]
fn should_reject_a_worker_replacement_when_the_worker_is_not_ready() {
    let rejection = reject_worker_replacement_request(false, false);
    assert!(
        matches!(rejection, Some(WorkerControlError::MissingActiveWorker)),
        "an idle but unready worker must reject the replacement with MissingActiveWorker"
    );
}

#[test]
fn should_report_the_busy_generation_rejection_before_the_unready_rejection() {
    // Both guard conditions hold: the original inlined arms checked the busy
    // generation first, and the shared unit preserves that precedence.
    let rejection = reject_worker_replacement_request(true, false);
    assert!(
        matches!(rejection, Some(WorkerControlError::GenerationBusy)),
        "a busy generation must win over the unready condition"
    );
}
