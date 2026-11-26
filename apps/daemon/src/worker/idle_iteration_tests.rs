//! Unit contracts for when the worker loop's idle control tick has work.

use super::IdleControlTickConditions;

fn idle_ready_worker_with_a_loaded_model() -> IdleControlTickConditions {
    IdleControlTickConditions {
        is_worker_running: true,
        is_ready: true,
        is_serving_a_request: false,
        has_loaded_model: true,
        has_pending_astro_ssd_prompt_cache_clear: false,
    }
}

#[test]
fn should_run_the_tick_for_an_idle_ready_worker_holding_a_model() {
    assert!(idle_ready_worker_with_a_loaded_model().are_met());
}

#[test]
fn should_run_the_tick_for_a_pending_clear_even_without_a_loaded_model() {
    let conditions = IdleControlTickConditions {
        has_loaded_model: false,
        has_pending_astro_ssd_prompt_cache_clear: true,
        ..idle_ready_worker_with_a_loaded_model()
    };

    assert!(conditions.are_met());
}

#[test]
fn should_skip_the_tick_when_there_is_nothing_to_sample_or_clear() {
    let conditions = IdleControlTickConditions {
        has_loaded_model: false,
        ..idle_ready_worker_with_a_loaded_model()
    };

    assert!(!conditions.are_met());
}

#[test]
fn should_skip_the_tick_while_a_request_is_being_served() {
    let conditions = IdleControlTickConditions {
        is_serving_a_request: true,
        ..idle_ready_worker_with_a_loaded_model()
    };

    assert!(!conditions.are_met());
}

#[test]
fn should_skip_the_tick_when_the_worker_is_not_running_or_not_ready() {
    let stopped_worker = IdleControlTickConditions {
        is_worker_running: false,
        ..idle_ready_worker_with_a_loaded_model()
    };
    let unready_worker = IdleControlTickConditions {
        is_ready: false,
        ..idle_ready_worker_with_a_loaded_model()
    };

    assert!(!stopped_worker.are_met());
    assert!(!unready_worker.are_met());
}
