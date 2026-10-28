//! Wall-clock watchdogs that turn the testing rules into physics.

use std::any::Any;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

/// Runs `body` on a worker thread joined under `timeout_seconds`; a hang
/// fails the test with a diagnostic naming it, and a panic inside the body
/// propagates unchanged so libtest reports it as an ordinary failure.
///
/// On timeout the worker thread is deliberately leaked: the Rust process
/// exits when the test binary's main thread finishes, reaping it then.
pub fn run_with_timeout(
    test_name: &str,
    timeout_seconds: u64,
    body: impl FnOnce() + Send + 'static,
) {
    let (result_sender, result_receiver) = mpsc::channel();
    thread::spawn(move || {
        let body_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(body));
        let _ = result_sender.send(body_result);
    });
    match result_receiver.recv_timeout(Duration::from_secs(timeout_seconds)) {
        Ok(Ok(())) => {}
        Ok(Err(panic_payload)) => resume_test_panic(panic_payload),
        Err(_) => panic!(
            "{test_name} exceeded its {}-second timeout and was aborted; \
             a test that needs longer must justify an explicit override \
             in the repository testing rules",
            timeout_seconds
        ),
    }
}

fn resume_test_panic(panic_payload: Box<dyn Any + Send>) {
    if let Some(message) = panic_payload.downcast_ref::<&str>() {
        panic!("test body panicked: {message}");
    }
    if let Some(message) = panic_payload.downcast_ref::<String>() {
        panic!("test body panicked: {message}");
    }
    panic!("test body panicked");
}

#[cfg(test)]
mod tests {
    //! The harness must obey its own rules: these tests pin the watchdog's
    //! pass, panic-propagation, and timeout-abort behavior with short budgets,
    //! plus the real-clock probe that guards against a frozen time source.
    //! They live beside `run_with_timeout` because each one exercises exactly
    //! that one unit with in-memory bodies and crosses no process, socket,
    //! filesystem, or model boundary of any kind.

    use super::run_with_timeout;

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
}
