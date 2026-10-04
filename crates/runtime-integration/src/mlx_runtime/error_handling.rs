//! Typed MLX runtime error classification for Astronomical policy.
//!
//! The captured-error machinery lives in `astronomical-mlx-c-rust`; this
//! module adds what only Astronomical knows: which descriptions name the
//! active memory ceiling enforced by the allocator patch, and how they become
//! typed capacity errors that the worker can recover from.

use std::sync::Mutex;

use crate::MlxRuntimeError;

pub fn classify_mlx_error(operation: &'static str, description: String) -> MlxRuntimeError {
    parse_active_memory_limit_error(&description).unwrap_or(MlxRuntimeError::RuntimeOperation {
        operation,
        description,
    })
}

pub(crate) fn check_status(status: i32, operation: &'static str) -> Result<(), MlxRuntimeError> {
    astronomical_mlx_c_rust::check_status(status, operation).map_err(MlxRuntimeError::from)
}

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
            _ => return None,
        }
    }
    Some(MlxRuntimeError::ActiveMemoryLimitExceeded {
        active_memory_bytes: active_memory_bytes?,
        attempted_allocation_bytes: attempted_allocation_bytes?,
        allowed_active_memory_bytes: allowed_active_memory_bytes?,
    })
}

pub(super) fn lock_unpoisoned<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}
