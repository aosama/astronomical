//! PyTorch convolution weights in the layout MLX convolutions read.
//!
//! PyTorch stores a convolution as `[out, in / groups, kernel]` and a
//! transposed convolution as `[in, out / groups, kernel]`; MLX reads both as
//! `[out, kernel, in / groups]`. Many checkpoints also store convolutions as a
//! weight-norm pair, `weight_g` and `weight_v`, folded here once at load.

use astronomical_runtime_integration::MlxRuntime;
use mlx_bridge::MlxArray;

use crate::speech_models::checkpoint_tensors::CheckpointTensors;
use crate::speech_models::speech_model_error::SpeechModelError;

/// `prefix.weight`, or the weight-norm pair `g * v / |v|` with the norm taken
/// over every axis but the first, as `torch.nn.utils.weight_norm` defines it.
/// Still in PyTorch's layout.
pub(crate) fn pytorch_convolution_weight(
    checkpoint: &CheckpointTensors<'_>,
    prefix: &str,
) -> Result<MlxArray, SpeechModelError> {
    if let Some(plain_weight) = checkpoint.optional_tensor(&format!("{prefix}.weight"))? {
        return Ok(plain_weight);
    }
    let direction = checkpoint.tensor(&format!("{prefix}.weight_v"))?;
    let magnitude = checkpoint.tensor(&format!("{prefix}.weight_g"))?;
    let runtime = checkpoint.runtime;
    let squared_direction = runtime.multiply(&direction, &direction)?;
    let squared_norm =
        runtime.sum_axis(&runtime.sum_axis(&squared_direction, 2, true)?, 1, true)?;
    let unit_direction = runtime.multiply(&direction, &runtime.rsqrt(&squared_norm)?)?;
    let folded_weight = runtime.multiply(&unit_direction, &magnitude)?;
    runtime.evaluate_arrays(&[&folded_weight])?;
    Ok(folded_weight)
}

/// `[out, in / groups, kernel]` to `[out, kernel, in / groups]`.
pub(crate) fn convolution_weight_for_mlx(
    runtime: &MlxRuntime,
    pytorch_weight: &MlxArray,
) -> Result<MlxArray, SpeechModelError> {
    let mlx_weight = runtime.transpose_axes(pytorch_weight, &[0, 2, 1])?;
    runtime.evaluate_arrays(&[&mlx_weight])?;
    Ok(mlx_weight)
}

/// `[in, out / groups, kernel]` to `[out, kernel, in / groups]`: each group
/// is `[in / groups, out / groups, kernel]`, reordered to
/// `[out / groups, kernel, in / groups]`, the groups stacked on `out`.
pub(crate) fn transposed_convolution_weight_for_mlx(
    runtime: &MlxRuntime,
    pytorch_weight: &MlxArray,
    groups: i32,
) -> Result<MlxArray, SpeechModelError> {
    let [input_channels, output_channels_per_group, kernel_size] = pytorch_weight.shape()[..]
    else {
        return Err(SpeechModelError::Artifact(format!(
            "a transposed convolution weight is {:?}, not [in, out, kernel]",
            pytorch_weight.shape()
        )));
    };
    let input_channels_per_group = input_channels / groups;
    let grouped = runtime.reshape(
        pytorch_weight,
        &[
            groups,
            input_channels_per_group,
            output_channels_per_group,
            kernel_size,
        ],
    )?;
    let reordered = runtime.transpose_axes(&grouped, &[0, 2, 3, 1])?;
    let mlx_weight = runtime.reshape(
        &reordered,
        &[
            groups * output_channels_per_group,
            kernel_size,
            input_channels_per_group,
        ],
    )?;
    runtime.evaluate_arrays(&[&mlx_weight])?;
    Ok(mlx_weight)
}

#[cfg(test)]
#[path = "convolution_weight_layout_tests.rs"]
mod tests;
