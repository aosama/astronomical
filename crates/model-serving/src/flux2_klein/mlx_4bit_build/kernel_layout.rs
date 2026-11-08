//! The one place that knows how the 4-bit build orders convolution kernels.
//!
//! mflux stores every VAE convolution kernel in MLX's `[out, kh, kw, in]`
//! order; the engine and the inventories use the Diffusers `[out, in, kh, kw]`
//! order. The rule applies to every 4-D VAE tensor and is never inferred from
//! a shape: `encoder.conv_in` (`[128, 3, 3, 3]`) reads the same either way.

/// Axes that turn an MLX-order kernel into Diffusers order.
#[cfg(feature = "direct-mlx")]
pub(crate) const MLX_TO_DIFFUSERS_KERNEL_AXES: [i32; 4] = [0, 3, 1, 2];

/// The Diffusers-order shape of an MLX-order kernel, or `None` when the shape
/// is not a 4-D kernel.
#[must_use]
pub(crate) fn diffusers_kernel_shape(mlx_kernel_shape: &[usize]) -> Option<Vec<usize>> {
    let &[output_channels, kernel_height, kernel_width, input_channels] = mlx_kernel_shape else {
        return None;
    };
    Some(vec![
        output_channels,
        input_channels,
        kernel_height,
        kernel_width,
    ])
}

#[cfg(test)]
mod tests {
    use super::diffusers_kernel_shape;

    #[test]
    fn should_move_input_channels_ahead_of_the_kernel_window() {
        assert_eq!(
            diffusers_kernel_shape(&[512, 3, 3, 32]),
            Some(vec![512, 32, 3, 3])
        );
    }

    #[test]
    fn should_leave_tensors_that_are_not_kernels_alone() {
        assert_eq!(diffusers_kernel_shape(&[512]), None);
        assert_eq!(diffusers_kernel_shape(&[512, 512]), None);
        assert_eq!(diffusers_kernel_shape(&[]), None);
    }
}
