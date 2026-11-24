//! A trainable head's parameters as one flat 32-bit vector with named parts.
//!
//! Keeping every trainable tensor in one vector lets the optimizer, the pull
//! back toward the shipped weights and the gradient checks treat a head as a
//! single array, while the layout still knows which slice is which tensor so
//! the result can be written under the shipped names.

use decision_contract::head_tuning_file::{HeadTuning, TunedTensor};

/// One named tensor's place in the flat vector.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParameterTensor {
    pub name: String,
    /// Empty for a scalar.
    pub shape: Vec<usize>,
    pub offset: usize,
    pub length: usize,
}

/// Where every trainable tensor sits in the flat vector.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParameterLayout {
    pub tensors: Vec<ParameterTensor>,
    pub parameter_count: usize,
}

impl ParameterLayout {
    /// Lays the named tensors out one after another, in the order given.
    #[must_use]
    pub fn new(named_shapes: &[(&str, Vec<usize>)]) -> Self {
        let mut tensors = Vec::with_capacity(named_shapes.len());
        let mut offset = 0;
        for (tensor_name, shape) in named_shapes {
            let mut length = 1;
            for dimension in shape {
                length *= dimension;
            }
            tensors.push(ParameterTensor {
                name: (*tensor_name).to_owned(),
                shape: shape.clone(),
                offset,
                length,
            });
            offset += length;
        }
        Self {
            tensors,
            parameter_count: offset,
        }
    }

    /// The tensor named `tensor_name`.
    ///
    /// # Panics
    /// When the layout has no such tensor: the family modules ask only for
    /// names they laid out themselves.
    #[must_use]
    pub fn tensor(&self, tensor_name: &str) -> &ParameterTensor {
        for parameter_tensor in &self.tensors {
            if parameter_tensor.name == tensor_name {
                return parameter_tensor;
            }
        }
        panic!("the layout has no tensor {tensor_name}");
    }

    /// The trained values as the tuning file stores them, one tensor per name.
    #[must_use]
    pub fn head_tuning(&self, parameters: &[f32]) -> HeadTuning {
        let mut tensors = Vec::with_capacity(self.tensors.len());
        for parameter_tensor in &self.tensors {
            let tensor_range =
                parameter_tensor.offset..parameter_tensor.offset + parameter_tensor.length;
            tensors.push(TunedTensor {
                name: parameter_tensor.name.clone(),
                shape: parameter_tensor.shape.clone(),
                values: parameters[tensor_range].to_vec(),
            });
        }
        HeadTuning { tensors }
    }
}

impl ParameterTensor {
    /// This tensor's values within the flat vector.
    #[must_use]
    pub fn values<'parameters>(&self, parameters: &'parameters [f32]) -> &'parameters [f32] {
        &parameters[self.offset..self.offset + self.length]
    }

    /// This tensor's values within the flat vector, to change.
    pub fn values_mut<'parameters>(
        &self,
        parameters: &'parameters mut [f32],
    ) -> &'parameters mut [f32] {
        &mut parameters[self.offset..self.offset + self.length]
    }
}

/// A weight's and its bias's gradient slices, which sit next to each other
/// in the flat layout (weight first).
pub fn split_weight_and_bias<'gradient>(
    gradient: &'gradient mut [f32],
    weight: &ParameterTensor,
    bias: &ParameterTensor,
) -> (&'gradient mut [f32], &'gradient mut [f32]) {
    debug_assert_eq!(weight.offset + weight.length, bias.offset);
    let weight_and_bias = &mut gradient[weight.offset..bias.offset + bias.length];
    weight_and_bias.split_at_mut(weight.length)
}

#[cfg(test)]
#[path = "head_parameters_tests.rs"]
mod tests;
