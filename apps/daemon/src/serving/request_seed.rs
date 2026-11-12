//! The seed a daemon picks for an image or speech request that did not carry
//! one.
//!
//! Every entry point (REST and the CLI's IPC verbs) needs the same rule, so it
//! lives with the serving layer, not in any surface. The seed is not secret:
//! it is reported back with every result, so a caller can reproduce the image
//! or the audio by passing it in. It only has to differ between requests.

use std::time::{SystemTime, UNIX_EPOCH};

/// Mixes the wall clock with the request identifier so two requests in the
/// same nanosecond, or after a clock reset, still draw different seeds. A
/// clock before the Unix epoch contributes nothing, leaving the identifier.
pub(crate) fn generated_seed(request_id: u64) -> u64 {
    let time_seed = match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(duration_since_epoch) => {
            duration_since_epoch.as_secs().rotate_left(32)
                ^ u64::from(duration_since_epoch.subsec_nanos())
        }
        Err(_clock_before_epoch) => 0,
    };
    time_seed ^ request_id.rotate_left(17)
}

#[cfg(test)]
#[path = "request_seed_tests.rs"]
mod request_seed_tests;
