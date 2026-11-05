//! What can go wrong in the runtime, and reading the bridge's error messages.

use std::ffi::{CStr, c_char};
use std::time::Duration;

use thiserror::Error;

pub(super) const NATIVE_ERROR_CAPACITY: usize = 512;

#[derive(Debug, Error)]
pub enum CoreMlProjectionError {
    #[error("could not place the compiled model files: {0}")]
    ModelFile(#[from] std::io::Error),
    #[error("the model path contains an interior NUL byte")]
    ModelPath,
    #[error("this program has no compiled model files to keep")]
    NoCompiledModelFiles,
    #[error("Core ML could not load the projection: {0}")]
    Load(String),
    #[error("expected {expected} fp16 values, got {actual}")]
    ElementCountMismatch { expected: usize, actual: usize },
    #[error("the IOSurface copy failed")]
    SurfaceCopy,
    #[error("the output IOSurface could not be locked for reading")]
    SurfaceLock,
    #[error("Core ML prediction failed: {0}")]
    Prediction(String),
    #[error("the asynchronous prediction could not start")]
    PredictionStart,
    #[error("the asynchronous prediction did not finish within {0:?}")]
    PredictionTimeout(Duration),
}

impl CoreMlProjectionError {
    pub(super) fn from_interior_nul(_interior_nul: std::ffi::NulError) -> Self {
        Self::ModelPath
    }
}

pub(super) fn native_error_text(native_error: &[c_char]) -> String {
    // SAFETY: the buffer is zero-initialized and the bridge always leaves its
    // last byte NUL, so a terminator exists inside the slice.
    let message = unsafe { CStr::from_ptr(native_error.as_ptr()) };
    message.to_string_lossy().into_owned()
}

#[cfg(test)]
#[path = "error_tests.rs"]
mod tests;
