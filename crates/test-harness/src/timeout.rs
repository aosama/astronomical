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
