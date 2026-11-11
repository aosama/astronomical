//! Element-wise activations the bridge does not provide.

use astronomical_runtime_integration::MlxRuntime;
use mlx_bridge::MlxArray;

use super::scalar_arithmetic::add_scalar;
use crate::speech_models::speech_model_error::SpeechModelError;

/// `max(x, slope * x)`.
pub(crate) fn leaky_relu(
    runtime: &MlxRuntime,
    input: &MlxArray,
    negative_slope: f32,
) -> Result<MlxArray, SpeechModelError> {
    Ok(runtime.maximum(input, &runtime.multiply_scalar(input, negative_slope)?)?)
}

/// ELU with alpha 1: `x` above zero, `exp(x) - 1` below. Written as
/// `max(x, 0) + exp(min(x, 0)) - 1`, which is the same function without a
/// select, and never exponentiates a large positive value.
pub(crate) fn elu(runtime: &MlxRuntime, input: &MlxArray) -> Result<MlxArray, SpeechModelError> {
    let zero = runtime.array_from_f32(&[0.0], &[])?;
    let positive_part = runtime.maximum(input, &zero)?;
    let negative_part = runtime.exp(&runtime.minimum(input, &zero)?)?;
    Ok(runtime.add(&positive_part, &add_scalar(runtime, &negative_part, -1.0)?)?)
}

/// `x * (1 + scale) + shift`: the adaptive modulation StyleTTS2's AdaIN and
/// the flow head's AdaLN both apply after normalising.
pub(crate) fn modulate(
    runtime: &MlxRuntime,
    normalized: &MlxArray,
    shift: &MlxArray,
    scale: &MlxArray,
) -> Result<MlxArray, SpeechModelError> {
    let one_plus_scale = add_scalar(runtime, scale, 1.0)?;
    Ok(runtime.add(&runtime.multiply(normalized, &one_plus_scale)?, shift)?)
}

#[cfg(test)]
#[path = "activations_tests.rs"]
mod tests;
