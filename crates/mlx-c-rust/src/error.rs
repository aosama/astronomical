//! Captured MLX-C error translation.
//!
//! MLX C reports an error immediately before returning nonzero on the same
//! calling thread, so the capture machinery pairs those two observations
//! thread-locally. This module only translates what MLX-C reported; typed
//! classification of Astronomical-specific error markers (for example the
//! active memory ceiling) is runtime policy and stays in
//! `astronomical-runtime-integration`, which converts `MlxCError` values at
//! its own boundary.
//!
//! Payload trampolines cannot return Rust errors across the C ABI, so they
//! park their typed failure in a separate thread-local slot and return
//! nonzero; status translation prefers that parked failure because it is the
//! root cause, while MLX-C's status text only reports that a callback errored.

use std::{
    cell::RefCell,
    ffi::{CStr, c_char, c_void},
    ptr,
    sync::Once,
};

use thiserror::Error;

use crate::raw;

static ERROR_HANDLER_INSTALLATION: Once = Once::new();

thread_local! {
    /// MLX C reports an error immediately before returning nonzero on the same
    /// calling thread. Thread-local storage preserves that operation pairing
    /// without allowing concurrent worker calls to consume each other's errors.
    static LAST_MLX_ERROR: RefCell<Option<String>> = const { RefCell::new(None) };
}

thread_local! {
    /// A failure raised inside a payload trampoline, parked for the caller's
    /// status translation. Routing it through MLX's own captured-error
    /// channel instead would let any nested MLX call on the same thread
    /// overwrite it before the caller observes the nonzero status, so the
    /// trampoline stores the typed failure here and returns nonzero.
    static LAST_CLOSURE_ERROR: RefCell<Option<MlxCError>> = const { RefCell::new(None) };
}

/// One failed MLX-C operation: what was attempted and what MLX-C reported.
#[derive(Debug, Error)]
#[error("MLX operation {operation} failed: {description}")]
pub struct MlxCError {
    /// The attempted operation, phrased the way runtime error reports name it.
    pub operation: &'static str,
    /// The description MLX-C reported, or a fallback when it reported none.
    pub description: String,
}

/// Installs the non-terminating MLX-C error handler exactly once per process.
///
/// The handler must be installed before any fallible MLX-C call so failures
/// carry descriptions instead of bare nonzero statuses.
pub fn install_non_terminating_error_handler() {
    ERROR_HANDLER_INSTALLATION.call_once(|| {
        // SAFETY: The callback follows MLX C's exact ABI, never unwinds, and
        // stores no borrowed pointers after returning. Null context and
        // destructor are valid because all state is Rust-owned static state.
        unsafe {
            raw::mlx_set_error_handler(Some(capture_mlx_error), ptr::null_mut(), None);
        }
    });
}

/// Translates an MLX-C status into `Ok(())` or a captured `MlxCError`.
///
/// # Errors
/// Returns the captured MLX-C description for the calling thread when the
/// status is nonzero, with a fallback description when MLX-C reported none.
pub fn check_status(status: i32, operation: &'static str) -> Result<(), MlxCError> {
    if status == 0 {
        clear_captured_mlx_error();
        clear_closure_error();
        return Ok(());
    }
    if let Some(failure) = take_closure_error() {
        // The parked trampoline failure is the root cause; the status-side
        // description is consumed too so it cannot leak into a later
        // operation's error on this thread.
        clear_captured_mlx_error();
        return Err(failure);
    }
    let description = take_captured_mlx_error()
        .unwrap_or_else(|| format!("MLX C returned status {status} without an error message"));
    Err(MlxCError {
        operation,
        description,
    })
}

/// Takes the captured MLX-C error description for the calling thread, if any.
#[must_use]
pub fn take_captured_mlx_error() -> Option<String> {
    LAST_MLX_ERROR.with(|last_error| {
        last_error
            .try_borrow_mut()
            .ok()
            .and_then(|mut writable_error| writable_error.take())
    })
}

/// Discards the captured MLX-C error description for the calling thread.
pub fn clear_captured_mlx_error() {
    LAST_MLX_ERROR.with(|last_error| {
        if let Ok(mut writable_error) = last_error.try_borrow_mut() {
            *writable_error = None;
        }
    });
}

/// Parks a payload-closure failure for the calling thread's status check.
pub fn set_closure_error(failure: MlxCError) {
    LAST_CLOSURE_ERROR.with(|last_failure| {
        if let Ok(mut writable_failure) = last_failure.try_borrow_mut() {
            *writable_failure = Some(failure);
        }
    });
}

/// Takes the parked payload-closure failure for the calling thread, if any.
#[must_use]
pub fn take_closure_error() -> Option<MlxCError> {
    LAST_CLOSURE_ERROR.with(|last_failure| {
        last_failure
            .try_borrow_mut()
            .ok()
            .and_then(|mut writable_failure| writable_failure.take())
    })
}

/// Discards the parked payload-closure failure for the calling thread.
pub fn clear_closure_error() {
    LAST_CLOSURE_ERROR.with(|last_failure| {
        if let Ok(mut writable_failure) = last_failure.try_borrow_mut() {
            *writable_failure = None;
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
