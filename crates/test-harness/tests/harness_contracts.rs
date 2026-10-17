//! The harness must obey its own rules: these tests pin the watchdog's
//! pass, panic-propagation, and timeout-abort behavior with short budgets.

use astronomical_test_harness::run_with_timeout;

#[test]
fn should_pass_a_body_that_finishes_within_the_budget() {
    run_with_timeout("finishing_body", 5, || {
        assert_eq!(2 + 2, 4);
    });
}

#[test]
fn should_propagate_a_panic_from_the_body() {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run_with_timeout("panicking_body", 5, || {
            panic!("the body's own failure message");
        });
    }));
    let panic_message = result
        .err()
        .map(|payload| {
            payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
                .expect("panic payload should be a string")
        })
        .expect("the run should have panicked");
    assert!(
        panic_message.contains("the body's own failure message"),
        "panic payload should carry the body's message, got: {panic_message}"
    );
}

#[test]
fn should_abort_a_hung_body_at_the_timeout() {
    let started = std::time::Instant::now();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run_with_timeout("hung_body", 1, || {
            std::thread::sleep(std::time::Duration::from_secs(30));
        });
    }));
    assert!(result.is_err(), "the hung body should have been aborted");
    assert!(
        started.elapsed() < std::time::Duration::from_secs(10),
        "the watchdog should abort at the 1-second budget, not wait for the body"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn real_clock_macro_runs_a_body_on_advancing_time() {
    let started_at = std::time::Instant::now();
    tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    let elapsed = started_at.elapsed();
    assert!(
        elapsed >= std::time::Duration::from_millis(4),
        "the real-clock macro must not freeze time: elapsed {elapsed:?}"
    );
}
