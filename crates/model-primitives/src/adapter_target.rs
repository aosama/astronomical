//! Where an add-on attaches: a linear layer's shipped name and its widths.
//!
//! The Qwen 3.5 backbone lists its targets and decision training fills them,
//! so the type sits here, below both (issue #2340).

use astronomical_decision_contract::DecisionFailure;
use astronomical_runtime_integration::MlxSafetensors;

/// A linear layer that may carry an add-on, with its widths.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdapterTarget {
    /// The shipped weight's prefix, the name before `.weight`.
    pub name: String,
    pub input_width: usize,
    pub output_width: usize,
}

impl AdapterTarget {
    /// The target of an affine-quantized linear layer, its widths read from
    /// the stored tensors: `weight` has one row per output and `scales` one
    /// column per group of inputs. Only the names are looked up; no weight
    /// bytes are read.
    pub fn of_quantized_linear(
        tensors: &MlxSafetensors,
        name: String,
        quantization_group_size: usize,
    ) -> Result<Self, DecisionFailure> {
        let (Ok(packed_weight), Ok(scales)) = (
            tensors.tensor(&format!("{name}.weight")),
            tensors.tensor(&format!("{name}.scales")),
        ) else {
            return Err(DecisionFailure::fatal_execution(&format!(
                "the model has no quantized layer {name}"
            )));
        };
        let (weight_shape, scales_shape) = (packed_weight.shape(), scales.shape());
        if weight_shape.len() != 2 || scales_shape.len() != 2 {
            return Err(DecisionFailure::fatal_execution(&format!(
                "the quantized layer {name} is not a matrix"
            )));
        }
        Ok(Self {
            output_width: weight_shape[0] as usize,
            input_width: scales_shape[1] as usize * quantization_group_size,
            name,
        })
    }
}
