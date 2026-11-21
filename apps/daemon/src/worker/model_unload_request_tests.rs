use crate::worker::handle::{ACTIVE_GENERATION_SLOTS, GENERATION_QUEUE_DEPTH};
use crate::worker::idle_model_unload::WorkerIdleness;
use crate::worker::model_unload_request::{
    ModelUnloadDecision, ModelUnloadOutcome, WorkerState, decide_model_unload,
};

const READY: WorkerState = WorkerState {
    is_running: true,
    is_ready: true,
};

fn idle_worker(has_loaded_model: bool) -> WorkerIdleness {
    WorkerIdleness {
        has_loaded_model,
        has_active_request: false,
        free_active_generation_permits: ACTIVE_GENERATION_SLOTS,
        free_generation_queue_permits: GENERATION_QUEUE_DEPTH,
    }
}

#[test]
fn should_unload_a_loaded_model_when_nothing_is_running() {
    assert_eq!(
        decide_model_unload(Some("Ornith-1.5-9B".to_owned()), READY, &idle_worker(true)),
        ModelUnloadDecision::Unload {
            model_id: "Ornith-1.5-9B".to_owned()
        }
    );
}

#[test]
fn should_answer_nothing_loaded_when_no_model_is_loaded() {
    assert_eq!(
        decide_model_unload(None, READY, &idle_worker(false)),
        ModelUnloadDecision::Answer(ModelUnloadOutcome::NothingLoaded)
    );
}

#[test]
fn should_refuse_while_a_request_is_being_served() {
    let mut serving = idle_worker(true);
    serving.has_active_request = true;
    assert_eq!(
        decide_model_unload(Some("Ornith-1.5-9B".to_owned()), READY, &serving),
        ModelUnloadDecision::Answer(ModelUnloadOutcome::Busy)
    );
}

#[test]
fn should_refuse_while_one_request_waits_for_admission() {
    let mut one_waiting = idle_worker(true);
    one_waiting.free_generation_queue_permits = GENERATION_QUEUE_DEPTH - 1;
    assert_eq!(
        decide_model_unload(Some("Ornith-1.5-9B".to_owned()), READY, &one_waiting),
        ModelUnloadDecision::Answer(ModelUnloadOutcome::Busy)
    );
}

#[test]
fn should_refuse_while_the_worker_is_still_loading_a_model() {
    assert_eq!(
        decide_model_unload(
            None,
            WorkerState {
                is_running: true,
                is_ready: false,
            },
            &idle_worker(false)
        ),
        ModelUnloadDecision::Answer(ModelUnloadOutcome::Busy)
    );
}

#[test]
fn should_answer_unavailable_when_the_worker_is_not_running() {
    assert_eq!(
        decide_model_unload(
            Some("Ornith-1.5-9B".to_owned()),
            WorkerState {
                is_running: false,
                is_ready: false,
            },
            &idle_worker(true)
        ),
        ModelUnloadDecision::Answer(ModelUnloadOutcome::Unavailable)
    );
}
