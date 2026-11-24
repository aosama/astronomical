//! The few dense operations the trainable heads need, on row-major `f32`
//! slices.
//!
//! The heads' trainable parts are a handful of linear layers, at most
//! 1024 × 4096. Plain loops over contiguous rows let the compiler vectorize
//! the inner products; a linear-algebra dependency would add more than it
//! saves at these sizes.

/// `output = weight · input + bias`, `weight` being `[output_width, input_width]`.
pub fn linear_forward(weight: &[f32], bias: &[f32], input: &[f32], output: &mut Vec<f32>) {
    output.clear();
    let input_width = input.len();
    for (weight_row, bias_value) in weight.chunks_exact(input_width).zip(bias) {
        output.push(dot(weight_row, input) + bias_value);
    }
}

/// The inner product of two equal-length slices.
#[must_use]
pub fn dot(left: &[f32], right: &[f32]) -> f32 {
    let mut accumulated = 0.0_f32;
    for (left_value, right_value) in left.iter().zip(right) {
        accumulated += left_value * right_value;
    }
    accumulated
}

/// Adds the gradients of `output = weight · input + bias` given
/// `output_gradient`: `weight_gradient += output_gradient ⊗ input` and
/// `bias_gradient += output_gradient`.
pub fn linear_parameter_gradients(
    input: &[f32],
    output_gradient: &[f32],
    weight_gradient: &mut [f32],
    bias_gradient: &mut [f32],
) {
    let input_width = input.len();
    for ((weight_gradient_row, bias_gradient_value), output_gradient_value) in weight_gradient
        .chunks_exact_mut(input_width)
        .zip(bias_gradient.iter_mut())
        .zip(output_gradient)
    {
        if *output_gradient_value == 0.0 {
            continue;
        }
        for (weight_gradient_value, input_value) in weight_gradient_row.iter_mut().zip(input) {
            *weight_gradient_value += output_gradient_value * input_value;
        }
        *bias_gradient_value += output_gradient_value;
    }
}

/// `input_gradient = weightᵀ · output_gradient`, the gradient flowing back
/// into a linear layer's input.
pub fn linear_input_gradient(
    weight: &[f32],
    output_gradient: &[f32],
    input_width: usize,
    input_gradient: &mut Vec<f32>,
) {
    input_gradient.clear();
    input_gradient.resize(input_width, 0.0);
    for (weight_row, output_gradient_value) in weight.chunks_exact(input_width).zip(output_gradient)
    {
        if *output_gradient_value == 0.0 {
            continue;
        }
        for (input_gradient_value, weight_value) in input_gradient.iter_mut().zip(weight_row) {
            *input_gradient_value += output_gradient_value * weight_value;
        }
    }
}

#[cfg(test)]
#[path = "dense_math_tests.rs"]
mod tests;
