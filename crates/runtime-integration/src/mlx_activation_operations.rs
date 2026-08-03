//! Activation and trigonometric graph operations used by model execution.
//!
//! Direct operations map to declarations in `mlx-c/mlx/c/ops.h` and forwarding
//! definitions in `mlx-c/mlx/c/ops.cpp`. Composite GELU functions intentionally
//! stay expressed as MLX graph operations so MLX controls dtype promotion,
//! kernel selection, and BF16 rounding; do not replace their internals with Rust
//! scalar math over copied device values.

use crate::{MlxArray, MlxRuntime, MlxRuntimeError, raw};

impl MlxRuntime {
    /// Applies numerically stable softplus through MLX `logaddexp(input, 0)`.
    ///
    /// This matches MLX's `nn.softplus`, which delegates to `logaddexp(x, 0)`.
    /// The naive `log1p(exp(x))` overflows to infinity for large positive
    /// inputs (x > ~88 in float32), which corrupts gated-delta decay values
    /// during long-context prefill.
    pub fn softplus(&self, input: &MlxArray) -> Result<MlxArray, MlxRuntimeError> {
        let zero_scalar = self.zeros(&[], input.dtype())?;
        self.logaddexp(input, &zero_scalar)
    }

    /// Applies the SiLU activation as `x * sigmoid(x)` on the MLX stream.
    pub fn silu(&self, input: &MlxArray) -> Result<MlxArray, MlxRuntimeError> {
        let sigmoid_weights = self.sigmoid(input)?;
        self.multiply(input, &sigmoid_weights)
    }

    /// Applies the elementwise tanh function on the MLX stream.
    pub fn tanh(&self, input: &MlxArray) -> Result<MlxArray, MlxRuntimeError> {
        self.output_array("apply MLX tanh", |output, stream| {
            // SAFETY: Input and stream are live and output is uniquely writable.
            unsafe { raw::mlx_tanh(output, input.raw(), stream) }
        })
    }

    /// Applies the elementwise error function through MLX-C `mlx_erf`.
    pub fn erf(&self, input: &MlxArray) -> Result<MlxArray, MlxRuntimeError> {
        self.output_array("apply MLX error function", |output, stream| {
            // SAFETY: Input and stream are live and output is uniquely writable.
            unsafe { raw::mlx_erf(output, input.raw(), stream) }
        })
    }

    /// Applies the elementwise cosine through MLX-C `mlx_cos`.
    pub fn cos(&self, input: &MlxArray) -> Result<MlxArray, MlxRuntimeError> {
        self.output_array("apply MLX cosine", |output, stream| {
            // SAFETY: Input and stream are live and output is uniquely writable.
            unsafe { raw::mlx_cos(output, input.raw(), stream) }
        })
    }

    /// Applies the elementwise sine through MLX-C `mlx_sin`.
    pub fn sin(&self, input: &MlxArray) -> Result<MlxArray, MlxRuntimeError> {
        self.output_array("apply MLX sine", |output, stream| {
            // SAFETY: Input and stream are live and output is uniquely writable.
            unsafe { raw::mlx_sin(output, input.raw(), stream) }
        })
    }

    /// Raises each broadcast-compatible base through MLX-C `mlx_power`.
    ///
    /// Using `mlx_power(x, 3)` rather than spelling `x*x*x` is deliberate. MLX's
    /// power kernel follows a different BF16 rounding path, and the vision GELU
    /// parity reference was generated with that operation.
    pub fn power(
        &self,
        bases: &MlxArray,
        exponents: &MlxArray,
    ) -> Result<MlxArray, MlxRuntimeError> {
        self.output_array("apply MLX power", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_power(output, bases.raw(), exponents.raw(), stream) }
        })
    }

    /// Applies exact GELU while preserving the activation dtype.
    ///
    /// Formula: `0.5*x*(1 + erf(x/sqrt(2)))`. MLX-C exposes `mlx_erf` but no
    /// single exact-GELU C entry point, so this composes `mlx_multiply`,
    /// `mlx_add`, and `mlx_erf`. The Qwen3.5 vision patch merger requires exact GELU.
    pub fn gelu(&self, input: &MlxArray) -> Result<MlxArray, MlxRuntimeError> {
        let scaled_input = self.multiply_scalar(input, std::f32::consts::FRAC_1_SQRT_2)?;
        let error_function_values = self.erf(&scaled_input)?;
        // Cast 1.0 to the input dtype before addition. Leaving it Float32 would
        // promote BF16 activations and change both memory use and numerical parity.
        let one_array_f32 = self.array_from_f32(&[1.0], &[1])?;
        let one_array = self.astype(&one_array_f32, input.dtype())?;
        let gaussian_cumulative_factor = self.add(&one_array, &error_function_values)?;
        let unscaled_activation = self.multiply(input, &gaussian_cumulative_factor)?;
        self.multiply_scalar(&unscaled_activation, 0.5)
    }

    /// Applies the GELU activation using the PyTorch tanh approximation.
    ///
    /// `GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`
    pub fn gelu_tanh(&self, input: &MlxArray) -> Result<MlxArray, MlxRuntimeError> {
        let sqrt_two_over_pi = 0.797_884_6_f32;
        let cubic_coefficient = 0.044715_f32;
        // The scalar exponent broadcasts over the input. Keep `x + 0.044715*x^3`
        // in this order; swapping coefficients or reassociating terms is not the
        // GELU approximation used by the checkpoint.
        let cubic_exponent = self.array_from_i32(&[3], &[])?;
        let cubed_input = self.power(input, &cubic_exponent)?;
        let weighted_cubed_input = self.multiply_scalar(&cubed_input, cubic_coefficient)?;
        let inner_polynomial = self.add(input, &weighted_cubed_input)?;
        let scaled_inner = self.multiply_scalar(&inner_polynomial, sqrt_two_over_pi)?;
        let tanh_result = self.tanh(&scaled_inner)?;
        // As in exact GELU, preserve BF16 by casting the additive identity first.
        let one_array_f32 = self.array_from_f32(&[1.0], &[1])?;
        let one_array = self.astype(&one_array_f32, input.dtype())?;
        let one_plus_tanh = self.add(&one_array, &tanh_result)?;
        let half_times_x = self.multiply_scalar(input, 0.5)?;
        self.multiply(&half_times_x, &one_plus_tanh)
    }
}
