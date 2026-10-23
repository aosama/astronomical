//! Typed MLX runtime error classification for Astronomical policy.
//!
//! The captured-error machinery lives in the published `mlx-c-rust` crate; this
//! module adds what only Astronomical knows: which descriptions name the
//! active memory ceiling enforced by the allocator patch, and how they become
//! typed capacity errors that the worker can recover from.
//!
//! # The marker contract
//!
//! The patched native allocator (see
//! `third-party/patches/mlx-0.32.4-astronomical-active-memory-ceiling.patch`)
//! rejects an allocation that would push active memory past the ceiling by
//! reporting an error whose description embeds the stable marker
//! `ASTRONOMICAL_MLX_ACTIVE_MEMORY_LIMIT_EXCEEDED` followed by three
//! `key=value` fields. Parsing that marker here is what converts a fatal
//! allocation failure into the recoverable
//! [`MlxRuntimeError::ActiveMemoryLimitExceeded`] that the engines' adaptive
//! prefill logic depends on. Lose this classification and a request that
//! should shrink its chunk and retry instead kills the worker.

use std::sync::Mutex;

use crate::MlxRuntimeError;

/// Classifies a captured MLX-C failure into the Astronomical error taxonomy.
///
/// The active-memory-ceiling marker is checked first; anything else becomes a
/// plain [`MlxRuntimeError::RuntimeOperation`] preserving the operation name
/// and the native description verbatim.
pub fn classify_mlx_error(operation: &'static str, description: String) -> MlxRuntimeError {
    parse_active_memory_limit_error(&description).unwrap_or(MlxRuntimeError::RuntimeOperation {
        operation,
        description,
    })
}

/// Status-code check shared by every fallible MLX-C call in this crate.
///
/// Delegates to the published crate's checker (which consults the captured
/// error channel installed at initialization) and funnels the result through
/// the `From` conversion so the classifier above sees every failure.
pub(crate) fn check_status(status: i32, operation: &'static str) -> Result<(), MlxRuntimeError> {
    mlx_c_rust::check_status(status, operation).map_err(MlxRuntimeError::from)
}

/// Extracts the typed active-memory-ceiling error from a native description.
///
/// Returns `None` for any description the parser does not fully understand —
/// a partial parse must degrade to the generic operation error rather than
/// report invented numbers, because the recovery arithmetic consumes these
/// byte values.
fn parse_active_memory_limit_error(description: &str) -> Option<MlxRuntimeError> {
    const ERROR_MARKER: &str = "ASTRONOMICAL_MLX_ACTIVE_MEMORY_LIMIT_EXCEEDED";
    // Native C and C++ boundaries add operation context before the shared
    // marker, for example "native MLX operation failed: <marker> ...". The
    // marker is the stable contract; requiring it at byte zero loses the typed
    // capacity classification and turns a recoverable request rejection into a
    // fatal worker failure.
    let marker_payload = description.split_once(ERROR_MARKER)?.1;
    let marker_fields =
        if let Some((marker_fields, native_location)) = marker_payload.split_once(" at ") {
            if native_location.is_empty() {
                return None;
            }
            marker_fields
        } else {
            marker_payload
        };
    let error_fields = marker_fields.split_whitespace();
    let mut active_memory_bytes = None;
    let mut attempted_allocation_bytes = None;
    let mut allowed_active_memory_bytes = None;
    for error_field in error_fields {
        let (field_name, field_text) = error_field.split_once('=')?;
        match field_name {
            "active_bytes" if active_memory_bytes.is_none() => {
                active_memory_bytes = field_text.parse().ok();
            }
            "allocation_bytes" if attempted_allocation_bytes.is_none() => {
                attempted_allocation_bytes = field_text.parse().ok();
            }
            "allowed_bytes" if allowed_active_memory_bytes.is_none() => {
                allowed_active_memory_bytes = field_text.parse().ok();
            }
            // An unknown field means the contract changed; refuse the parse
            // instead of guessing which numbers mean what.
            _ => return None,
        }
    }
    Some(MlxRuntimeError::ActiveMemoryLimitExceeded {
        active_memory_bytes: active_memory_bytes?,
        attempted_allocation_bytes: attempted_allocation_bytes?,
        allowed_active_memory_bytes: allowed_active_memory_bytes?,
    })
}

/// Locks a mutex while tolerating a poisoned state.
///
/// Every mutex in this module guards process-global MLX policy, and a panic
/// while holding one leaves the *native* state intact and knowable — the
/// poison only means a Rust-side observer bailed mid-read. Recovering the
/// lock beats propagating the poison, which would cascade into unrelated
/// operations failing forever after one panic.
pub(super) fn lock_unpoisoned<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}
