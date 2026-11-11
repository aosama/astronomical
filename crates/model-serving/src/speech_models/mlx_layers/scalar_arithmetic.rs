//! Arithmetic with one scalar broadcast over an array, which the bridge
//! offers for multiplication only.

use astronomical_runtime_integration::MlxRuntime;
use mlx_bridge::MlxArray;

use crate::speech_models::speech_model_error::SpeechModelError;

/// `x + addend` on every element.
pub(crate) fn add_scalar(
    runtime: &MlxRuntime,
    input: &MlxArray,
    addend: f32,
) -> Result<MlxArray, SpeechModelError> {
    Ok(runtime.add(input, &runtime.array_from_f32(&[addend], &[])?)?)
}
