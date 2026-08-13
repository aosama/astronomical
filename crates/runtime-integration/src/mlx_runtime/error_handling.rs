use std::{
    cell::RefCell,
    ffi::{CStr, c_char, c_void},
    ptr,
    sync::{Mutex, Once},
};

use crate::{MlxRuntimeError, raw};

static ERROR_HANDLER_INSTALLATION: Once = Once::new();

thread_local! {
    /// MLX C reports an error immediately before returning nonzero on the same
    /// calling thread. Thread-local storage preserves that operation pairing
    /// without allowing concurrent worker calls to consume each other's errors.
    static LAST_MLX_ERROR: RefCell<Option<String>> = const { RefCell::new(None) };
}

pub(crate) fn install_non_terminating_error_handler() {
    ERROR_HANDLER_INSTALLATION.call_once(|| {
        // SAFETY: The callback follows MLX C's exact ABI, never unwinds, and
        // stores no borrowed pointers after returning. Null context and
        // destructor are valid because all state is Rust-owned static state.
        unsafe {
            raw::mlx_set_error_handler(Some(capture_mlx_error), ptr::null_mut(), None);
        }
    });
}

unsafe extern "C" fn capture_mlx_error(message: *const c_char, _context: *mut c_void) {
    let description = if message.is_null() {
        "MLX reported an error without a message".to_owned()
    } else {
        // SAFETY: MLX C documents that the callback receives a valid
        // null-terminated message for the duration of this call.
        unsafe { CStr::from_ptr(message) }
            .to_string_lossy()
            .into_owned()
    };
    LAST_MLX_ERROR.with(|last_error| {
        if let Ok(mut writable_error) = last_error.try_borrow_mut() {
            *writable_error = Some(description);
        }
    });
}

pub(crate) fn check_status(status: i32, operation: &'static str) -> Result<(), MlxRuntimeError> {
    if status == 0 {
        clear_captured_mlx_error();
        return Ok(());
    }
    let description = take_captured_mlx_error()
        .unwrap_or_else(|| format!("MLX C returned status {status} without an error message"));
    Err(classify_mlx_error(operation, description))
}

pub fn classify_mlx_error(operation: &'static str, description: String) -> MlxRuntimeError {
    parse_active_memory_limit_error(&description).unwrap_or(MlxRuntimeError::RuntimeOperation {
        operation,
        description,
    })
}

pub(crate) fn take_captured_mlx_error() -> Option<String> {
    LAST_MLX_ERROR.with(|last_error| {
        last_error
            .try_borrow_mut()
            .ok()
            .and_then(|mut writable_error| writable_error.take())
    })
}

pub(crate) fn clear_captured_mlx_error() {
    LAST_MLX_ERROR.with(|last_error| {
        if let Ok(mut writable_error) = last_error.try_borrow_mut() {
            *writable_error = None;
        }
    });
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
