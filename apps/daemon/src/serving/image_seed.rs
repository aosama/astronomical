//! The seed a daemon picks for an image request that did not carry one.
//!
//! Both image entry points (REST and the CLI's IPC verb) need the same rule,
//! so it lives with the serving layer, not in either surface. The seed is not
//! secret: it is reported back in every completion, so a caller can reproduce
//! the image by passing it in. It only has to differ between requests.

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
#[path = "image_seed_tests.rs"]
mod image_seed_tests;
