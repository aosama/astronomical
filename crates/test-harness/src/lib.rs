//! Shared test harness: deterministic-time and timeout scaffolding, plus
//! workspace binary resolution, for the repository's test tiers.
//!
//! Vocabulary (issue #1208): **offline** names the property — sealed and
//! deterministic, no real network beyond localhost, no real model artifacts,
//! everything scripted. Scope words name the seam: **unit** (in-process,
//! fakes), **integration** (stubs or fixture binaries), **end-to-end (E2E)**
//! (the real `astronomicald` binary over real HTTP), **acceptance** (real
//! model artifacts).
//!
//! The harness turns the repository's standing testing rules into physics:
//! every test declared through these macros carries the default 120-second
//! timeout without anyone remembering to write one.

pub mod timeout;
pub mod workspace;

pub use timeout::run_with_timeout;
pub use workspace::workspace_binary;

/// Default test timeout in seconds, per the repository testing rules.
pub const DEFAULT_TEST_TIMEOUT_SECONDS: u64 = 120;

/// Declares a synchronous offline test with the default timeout enforced.
///
/// Expansion is a plain `#[test]` whose body runs on a worker thread joined
/// under the timeout; a hang fails the test with a diagnostic naming it, and
/// a panic inside the body propagates unchanged.
#[macro_export]
macro_rules! offline_test {
    ($name:ident, $body:block) => {
        #[test]
        fn $name() {
            $crate::run_with_timeout(
                stringify!($name),
                $crate::DEFAULT_TEST_TIMEOUT_SECONDS,
                || $body,
            );
        }
    };
    ($name:ident, $timeout_seconds:expr, $body:block) => {
        #[test]
        fn $name() {
            $crate::run_with_timeout(stringify!($name), $timeout_seconds, || $body);
        }
    };
}

/// Declares an asynchronous offline test on Tokio's paused (virtual) clock
/// with the default timeout enforced in real wall-clock time.
///
/// Paused time makes every `tokio::time::sleep` in the body advance instantly
/// and deterministically — virtual time is the default here, and a body that
/// finishes only by waiting on real elapsed time will trip the wall-clock
/// timeout instead. The consumer crate needs `tokio` among its
/// dev-dependencies; the expansion is self-contained otherwise.
#[macro_export]
macro_rules! offline_tokio_test {
    ($name:ident, $body:block) => {
        #[tokio::test(flavor = "current_thread", start_paused = true)]
        async fn $name() {
            if ::tokio::time::timeout(
                ::std::time::Duration::from_secs($crate::DEFAULT_TEST_TIMEOUT_SECONDS),
                async { $body },
            )
            .await
            .is_err()
            {
                panic!(
                    "{} exceeded its {}-second timeout and was aborted",
                    stringify!($name),
                    $crate::DEFAULT_TEST_TIMEOUT_SECONDS
                );
            }
        }
    };
    ($name:ident, $timeout_seconds:expr, $body:block) => {
        #[tokio::test(flavor = "current_thread", start_paused = true)]
        async fn $name() {
            if ::tokio::time::timeout(::std::time::Duration::from_secs($timeout_seconds), async {
                $body
            })
            .await
            .is_err()
            {
                panic!(
                    "{} exceeded its {}-second timeout and was aborted",
                    stringify!($name),
                    $timeout_seconds
                );
            }
        }
    };
}

/// Declares an asynchronous integration test on Tokio's real clock with the
/// default timeout enforced in real wall-clock time.
///
/// Use this only for tests whose subject genuinely requires real elapsed time
/// alongside real I/O — for example unix-socket transport journeys whose
/// `tokio::time::timeout` guards wrap real peer reads. A paused clock never
/// advances on real I/O, so such guards cannot fire there; this macro keeps
/// the harness-enforced timeout without freezing time. Deterministic bodies
/// belong in `offline_tokio_test!`, not here.
#[macro_export]
macro_rules! offline_tokio_real_clock_test {
    ($name:ident, $body:block) => {
        #[tokio::test(flavor = "current_thread")]
        async fn $name() {
            if ::tokio::time::timeout(
                ::std::time::Duration::from_secs($crate::DEFAULT_TEST_TIMEOUT_SECONDS),
                async { $body },
            )
            .await
            .is_err()
            {
                panic!(
                    "{} exceeded its {}-second timeout and was aborted",
                    stringify!($name),
                    $crate::DEFAULT_TEST_TIMEOUT_SECONDS
                );
            }
        }
    };
}
