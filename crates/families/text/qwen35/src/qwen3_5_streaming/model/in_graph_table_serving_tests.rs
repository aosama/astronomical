//! How in-graph attempts are spaced after a refusal.

use std::sync::Arc;

use super::{
    InGraphLayerOutcome, InGraphLayerServing, InGraphTableServing,
    MAXIMUM_EXACT_TOKENS_AFTER_A_REFUSAL, SharedLayerOutcome, TableRefusal,
};
use crate::qwen3_5_streaming::expert_streaming::TableReadThrough;

/// How many tokens in a row wait before the next attempt is allowed.
fn exact_tokens_waited(in_graph_serving: &InGraphTableServing) -> u32 {
    let mut waited_token_count = 0;
    while in_graph_serving.waits_out_a_refusal() {
        waited_token_count += 1;
    }
    waited_token_count
}

#[test]
fn should_attempt_every_token_while_none_is_refused() {
    let in_graph_serving = InGraphTableServing::default();

    for _ in 0..3 {
        assert!(!in_graph_serving.waits_out_a_refusal());
        in_graph_serving.forget_refusals();
    }
}

#[test]
fn should_wait_twice_as_long_after_each_refusal_in_a_row_up_to_the_cap() {
    let in_graph_serving = InGraphTableServing::default();
    let mut waits = Vec::new();

    for _ in 0..8 {
        in_graph_serving.space_out_attempts_after_a_refusal();
        waits.push(exact_tokens_waited(&in_graph_serving));
    }

    assert_eq!(
        waits,
        vec![1, 2, 4, 8, 16, 32, 32, 32],
        "the wait doubles and stops at the cap"
    );
    assert_eq!(waits[5], MAXIMUM_EXACT_TOKENS_AFTER_A_REFUSAL);
}

#[test]
fn should_wait_one_token_again_after_a_served_token_ends_a_run_of_refusals() {
    let in_graph_serving = InGraphTableServing::default();
    for _ in 0..4 {
        in_graph_serving.space_out_attempts_after_a_refusal();
        exact_tokens_waited(&in_graph_serving);
    }

    in_graph_serving.forget_refusals();
    in_graph_serving.space_out_attempts_after_a_refusal();

    assert_eq!(exact_tokens_waited(&in_graph_serving), 1);
}

#[test]
fn should_serve_no_forward_and_report_no_refusal_before_a_forward_begins() {
    let in_graph_serving = InGraphTableServing::default();

    assert!(!in_graph_serving.is_serving_forward());
    assert!(!in_graph_serving.forward_was_refused());
    assert_eq!(in_graph_serving.counts().attempted_token_count, 0);
}

/// Adds a layer to the forward in progress with the outcome its lookup left.
fn add_layer_with_outcome(in_graph_serving: &InGraphTableServing, outcome: InGraphLayerOutcome) {
    let shared_outcome = Arc::new(SharedLayerOutcome::default());
    *shared_outcome.lock() = outcome;
    in_graph_serving
        .layer_servings
        .borrow_mut()
        .push(InGraphLayerServing {
            layer_index: 0,
            assignment_count: 8,
            table_payload_bytes: 0,
            expert_capacity: 256,
            outcome: shared_outcome,
        });
}

#[test]
fn should_serve_a_forward_whose_lookups_all_served() {
    let in_graph_serving = InGraphTableServing::default();
    in_graph_serving.window.open();
    add_layer_with_outcome(
        &in_graph_serving,
        InGraphLayerOutcome::Served(TableReadThrough::default()),
    );

    assert!(
        !in_graph_serving
            .conclude_forward()
            .expect("every lookup ran")
    );

    assert_eq!(in_graph_serving.counts().refused_token_count, 0);
    assert_eq!(
        in_graph_serving.layer_servings.borrow().len(),
        1,
        "the servings are kept for recording"
    );
}

#[test]
fn should_serve_a_refused_token_again_and_not_refuse_its_exact_rerun() {
    let in_graph_serving = InGraphTableServing::default();
    in_graph_serving.window.open();
    add_layer_with_outcome(
        &in_graph_serving,
        InGraphLayerOutcome::Refused(Some(TableRefusal::WouldGrow)),
    );
    in_graph_serving.window.mark_refused();

    assert!(
        in_graph_serving
            .conclude_forward()
            .expect("a refusal is not an error")
    );
    // The forward ends; the token is run again the exact way, and that
    // forward is concluded too, with the window closed and no lookups.
    in_graph_serving.window.close();
    let exact_rerun_was_refused = in_graph_serving.conclude_forward().expect("no lookups");

    assert!(
        !exact_rerun_was_refused,
        "an exact forward has no lookup to refuse it"
    );
    let counts = in_graph_serving.counts();
    assert_eq!(counts.refused_token_count, 1);
    assert_eq!(counts.would_grow_refusal_count, 1);
    assert!(in_graph_serving.layer_servings.borrow().is_empty());
    assert!(
        in_graph_serving.waits_out_a_refusal(),
        "the token after a refusal is served exactly"
    );
}

#[test]
fn should_fail_a_forward_nobody_refused_whose_lookup_never_ran() {
    let in_graph_serving = InGraphTableServing::default();
    in_graph_serving.window.open();
    add_layer_with_outcome(&in_graph_serving, InGraphLayerOutcome::NotRun);

    assert!(in_graph_serving.conclude_forward().is_err());
}

#[test]
fn should_not_fail_the_next_exact_forward_after_an_in_graph_forward_failed() {
    let in_graph_serving = InGraphTableServing::default();
    in_graph_serving.window.open();
    // The evaluation failed before this lookup ran, so nothing concluded it.
    add_layer_with_outcome(&in_graph_serving, InGraphLayerOutcome::NotRun);

    in_graph_serving.end_forward();

    assert!(in_graph_serving.layer_servings.borrow().is_empty());
    assert!(
        !in_graph_serving
            .conclude_forward()
            .expect("the exact forward has no lookups of its own"),
        "an exact forward is not refused"
    );
}

#[test]
fn should_count_a_refusal_for_want_of_seats_by_its_own_reason() {
    let in_graph_serving = InGraphTableServing::default();
    in_graph_serving.window.open();
    add_layer_with_outcome(
        &in_graph_serving,
        InGraphLayerOutcome::Refused(Some(TableRefusal::CannotSeatEveryExpert)),
    );
    add_layer_with_outcome(&in_graph_serving, InGraphLayerOutcome::Refused(None));
    in_graph_serving.window.mark_refused();

    assert!(
        in_graph_serving
            .conclude_forward()
            .expect("a refusal is not an error")
    );

    let counts = in_graph_serving.counts();
    assert_eq!(counts.cannot_seat_refusal_count, 1);
    assert_eq!(counts.would_grow_refusal_count, 0);
    assert_eq!(
        counts.refused_token_count, 1,
        "one token, however many layers"
    );
}
