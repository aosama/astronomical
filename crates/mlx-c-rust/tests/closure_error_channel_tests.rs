//! Hermetic contracts for the payload-closure error channel: the
//! thread-local parking lot that carries closure failures from C
//! trampolines back to the caller's status translation.

use astronomical_mlx_c_rust::MlxCError;
use astronomical_mlx_c_rust::error::{clear_closure_error, set_closure_error, take_closure_error};

fn parked_failure(operation: &'static str, description: &str) -> MlxCError {
    MlxCError {
        operation,
        description: description.to_owned(),
    }
}

#[test]
fn should_take_a_parked_failure_exactly_once() {
    clear_closure_error();
    set_closure_error(parked_failure("apply a Rust closure", "first failure"));

    let taken_failure = take_closure_error();

    let failure = taken_failure.expect("a parked failure must be retrievable once");
    assert_eq!(failure.operation, "apply a Rust closure");
    assert_eq!(failure.description, "first failure");
    assert!(
        take_closure_error().is_none(),
        "taking must clear the channel for the next operation"
    );
}

#[test]
fn should_keep_the_latest_failure_when_a_new_one_is_parked() {
    clear_closure_error();
    set_closure_error(parked_failure("apply a Rust closure", "outer failure"));
    set_closure_error(parked_failure(
        "apply a Rust kwargs closure",
        "inner failure",
    ));

    let failure = take_closure_error().expect("the channel keeps the latest failure");

    assert_eq!(
        failure.operation, "apply a Rust kwargs closure",
        "an inner closure failure must not be masked by the outer one"
    );
    assert_eq!(failure.description, "inner failure");
}

#[test]
fn should_clear_a_parked_failure_without_returning_it() {
    clear_closure_error();
    set_closure_error(parked_failure("apply a Rust closure", "stale failure"));

    clear_closure_error();

    assert!(
        take_closure_error().is_none(),
        "a cleared channel must report no failure"
    );
}

#[test]
fn should_report_no_failure_on_an_untouched_channel() {
    clear_closure_error();

    assert!(take_closure_error().is_none());
}
