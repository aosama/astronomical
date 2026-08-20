//! Thin validated wrappers over MLX-C convolution operations.
//!
//! C declarations: `mlx-c/mlx/c/ops.h::{mlx_conv1d, mlx_conv2d, mlx_conv3d}`.
//! C++ forwarding definitions: `mlx-c/mlx/c/ops.cpp`. MLX uses channel-last
//! inputs: Conv1d `[N,L,C]`, Conv2d `[N,H,W,C]`, and Conv3d `[N,D,H,W,C]`.
//! Weights begin with output channels and end with input channels per group.
//! These layouts differ from PyTorch.

use crate::{MlxArray, MlxRuntime, MlxRuntimeError, raw};

impl MlxRuntime {
    /// Applies MLX one-dimensional convolution over `[batch, length, channels]` inputs.
    pub fn conv1d(
        &self,
        input: &MlxArray,
        weight: &MlxArray,
        stride: i32,
        padding: i32,
        dilation: i32,
        groups: i32,
    ) -> Result<MlxArray, MlxRuntimeError> {
        validate_convolution_arguments(input, weight, &[stride], &[padding], &[dilation], groups)?;
        self.output_array("apply MLX conv1d", |output, stream| {
            // SAFETY: Arrays and stream are live and validated scalar arguments
            // match the official one-dimensional convolution contract.
            unsafe {
                raw::mlx_conv1d(
                    output,
                    input.raw(),
                    weight.raw(),
                    stride,
                    padding,
                    dilation,
                    groups,
                    stream,
                )
            }
        })
    }

    /// Applies MLX two-dimensional convolution over `[batch, height, width, channels]`.
    pub fn conv2d(
        &self,
        input: &MlxArray,
        weight: &MlxArray,
        strides: [i32; 2],
        paddings: [i32; 2],
        dilations: [i32; 2],
        groups: i32,
    ) -> Result<MlxArray, MlxRuntimeError> {
        validate_convolution_arguments(input, weight, &strides, &paddings, &dilations, groups)?;
        self.output_array("apply MLX conv2d", |output, stream| {
            // SAFETY: Arrays and stream are live and all convolution geometry
            // was validated against the official channel-last contract.
            unsafe {
                raw::mlx_conv2d(
                    output,
                    input.raw(),
                    weight.raw(),
                    strides[0],
                    strides[1],
                    paddings[0],
                    paddings[1],
                    dilations[0],
                    dilations[1],
                    groups,
                    stream,
                )
            }
        })
    }

    /// Applies MLX-C `mlx_conv3d` over `[batch, depth, height, width, channels]`.
    ///
    /// The Qwen3.5-MoE vision tower uses kernel=stride=`[temporal, patch, patch]`, so
    /// this operation projects disjoint patch volumes rather than sliding over
    /// neighboring pixels.
    pub fn conv3d(
        &self,
        input: &MlxArray,
        weight: &MlxArray,
        strides: [i32; 3],
        paddings: [i32; 3],
        dilations: [i32; 3],
        groups: i32,
    ) -> Result<MlxArray, MlxRuntimeError> {
        validate_convolution_arguments(input, weight, &strides, &paddings, &dilations, groups)?;
        self.output_array("apply MLX conv3d", |output, stream| {
            // SAFETY: Arrays and stream are live and validated scalar arguments
            // match the official three-dimensional convolution contract.
            unsafe {
                raw::mlx_conv3d(
                    output,
                    input.raw(),
                    weight.raw(),
                    strides[0],
                    strides[1],
                    strides[2],
                    paddings[0],
                    paddings[1],
                    paddings[2],
                    dilations[0],
                    dilations[1],
                    dilations[2],
                    groups,
                    stream,
                )
            }
        })
    }
}

fn validate_convolution_arguments(
    input: &MlxArray,
    weight: &MlxArray,
    strides: &[i32],
    paddings: &[i32],
    dilations: &[i32],
    groups: i32,
) -> Result<(), MlxRuntimeError> {
    const OPERATION: &str = "apply MLX convolution";
    // MLX convolution rank is batch + spatial axes + channel. Validate this at
    // the safe Rust boundary rather than relying on a C++ exception/status later.
    let expected_rank = strides.len() + 2;
    let input_shape = input.shape();
    let weight_shape = weight.shape();
    if !matches!(strides.len(), 1..=3)
        || paddings.len() != strides.len()
        || dilations.len() != strides.len()
        || input_shape.len() != expected_rank
        || weight_shape.len() != expected_rank
    {
        return Err(runtime_operation_error(
            OPERATION,
            "convolution arrays and spatial arguments must have matching ranks",
        ));
    }
    if strides.iter().any(|stride| *stride <= 0)
        || dilations.iter().any(|dilation| *dilation <= 0)
        || groups <= 0
    {
        return Err(runtime_operation_error(
            OPERATION,
            "convolution strides, dilations, and groups must be positive",
        ));
    }
    if paddings.iter().any(|padding| *padding < 0) {
        return Err(runtime_operation_error(
            OPERATION,
            "convolution paddings must be nonnegative",
        ));
    }
    if weight_shape[1..expected_rank - 1]
        .iter()
        .any(|kernel_size| *kernel_size <= 0)
    {
        return Err(runtime_operation_error(
            OPERATION,
            "convolution kernel dimensions must be positive",
        ));
    }
    let input_channel_count = input_shape[expected_rank - 1];
    let output_channel_count = weight_shape[0];
    let weight_channel_count = weight_shape[expected_rank - 1];
    if input_channel_count <= 0 || output_channel_count <= 0 || weight_channel_count <= 0 {
        return Err(runtime_operation_error(
            OPERATION,
            "convolution channel dimensions must be positive",
        ));
    }
    if input_channel_count % groups != 0 || output_channel_count % groups != 0 {
        return Err(runtime_operation_error(
            OPERATION,
            "convolution input and output channels must be divisible by groups",
        ));
    }
    if weight_channel_count != input_channel_count / groups {
        return Err(runtime_operation_error(
            OPERATION,
            "convolution weight channels must equal input channels per group",
        ));
    }
    Ok(())
}

fn runtime_operation_error(operation: &'static str, description: &'static str) -> MlxRuntimeError {
    MlxRuntimeError::RuntimeOperation {
        operation,
        description: description.to_owned(),
    }
}
