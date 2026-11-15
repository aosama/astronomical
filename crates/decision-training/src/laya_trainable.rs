//! Laya's scorer as a trainable part: `LayerNorm → Linear → GELU → Linear(1)`
//! over each option's marker state.
//!
//! The marker states come from the head's two transformer layers, which stay
//! frozen in this stage (they are stored 8-bit and need gradients through
//! attention). The normalization's input never changes during training, so
//! its normalized vector is computed once per option and only its scale and
//! shift train.

use astronomical_decision_contract::HeadInputs;

use crate::dense_math::{dot, linear_forward, linear_input_gradient, linear_parameter_gradients};
use crate::gelu::{gelu, gelu_derivative};
use crate::head_parameters::{ParameterLayout, ParameterTensor, split_weight_and_bias};
use crate::trainable_head::{HeadInputsMismatch, TrainableHead, input_family_name};

/// The scorer's normalization epsilon: PyTorch's default, which the
/// checkpoints were built with and serving uses.
const SCORER_LAYER_NORM_EPSILON: f64 = 1e-5;

/// Laya's scorer for one marker width and hidden width.
pub struct LayaTrainableHead {
    layout: ParameterLayout,
    marker_width: usize,
    hidden_width: usize,
    norm_scale: ParameterTensor,
    norm_shift: ParameterTensor,
    hidden_weight: ParameterTensor,
    hidden_bias: ParameterTensor,
    output_weight: ParameterTensor,
    output_bias: ParameterTensor,
}

/// One option's values on the way through the scorer.
struct ScorerPass {
    normalized: Vec<f32>,
    norm_output: Vec<f32>,
    hidden_input: Vec<f32>,
    hidden_activation: Vec<f32>,
    logit: f32,
}

impl LayaTrainableHead {
    #[must_use]
    pub fn new(marker_width: usize, hidden_width: usize) -> Self {
        let layout = ParameterLayout::new(&[
            ("scorer.layers.0.weight", vec![marker_width]),
            ("scorer.layers.0.bias", vec![marker_width]),
            ("scorer.layers.1.weight", vec![hidden_width, marker_width]),
            ("scorer.layers.1.bias", vec![hidden_width]),
            ("scorer.layers.3.weight", vec![1, hidden_width]),
            ("scorer.layers.3.bias", vec![1]),
        ]);
        Self {
            norm_scale: layout.tensor("scorer.layers.0.weight").clone(),
            norm_shift: layout.tensor("scorer.layers.0.bias").clone(),
            hidden_weight: layout.tensor("scorer.layers.1.weight").clone(),
            hidden_bias: layout.tensor("scorer.layers.1.bias").clone(),
            output_weight: layout.tensor("scorer.layers.3.weight").clone(),
            output_bias: layout.tensor("scorer.layers.3.bias").clone(),
            layout,
            marker_width,
            hidden_width,
        }
    }

    fn scorer_pass(&self, parameters: &[f32], marker_state: &[f32]) -> ScorerPass {
        let normalized = normalized_vector(marker_state);
        let mut norm_output = Vec::with_capacity(self.marker_width);
        for ((normalized_value, scale), shift) in normalized
            .iter()
            .zip(self.norm_scale.values(parameters))
            .zip(self.norm_shift.values(parameters))
        {
            norm_output.push(normalized_value * scale + shift);
        }
        let mut hidden_input = Vec::with_capacity(self.hidden_width);
        linear_forward(
            self.hidden_weight.values(parameters),
            self.hidden_bias.values(parameters),
            &norm_output,
            &mut hidden_input,
        );
        let mut hidden_activation = Vec::with_capacity(self.hidden_width);
        for hidden_value in &hidden_input {
            hidden_activation.push(gelu(*hidden_value));
        }
        let logit = dot(self.output_weight.values(parameters), &hidden_activation)
            + self.output_bias.values(parameters)[0];
        ScorerPass {
            normalized,
            norm_output,
            hidden_input,
            hidden_activation,
            logit,
        }
    }

    fn add_option_gradient(
        &self,
        parameters: &[f32],
        scorer_pass: &ScorerPass,
        logit_gradient: f32,
        gradient: &mut [f32],
    ) {
        let (output_weight_gradient, output_bias_gradient) =
            split_weight_and_bias(gradient, &self.output_weight, &self.output_bias);
        linear_parameter_gradients(
            &scorer_pass.hidden_activation,
            &[logit_gradient],
            output_weight_gradient,
            output_bias_gradient,
        );
        let mut hidden_input_gradient = Vec::with_capacity(self.hidden_width);
        for (output_weight_value, hidden_value) in self
            .output_weight
            .values(parameters)
            .iter()
            .zip(&scorer_pass.hidden_input)
        {
            hidden_input_gradient
                .push(logit_gradient * output_weight_value * gelu_derivative(*hidden_value));
        }
        let (hidden_weight_gradient, hidden_bias_gradient) =
            split_weight_and_bias(gradient, &self.hidden_weight, &self.hidden_bias);
        linear_parameter_gradients(
            &scorer_pass.norm_output,
            &hidden_input_gradient,
            hidden_weight_gradient,
            hidden_bias_gradient,
        );
        let mut norm_output_gradient = Vec::with_capacity(self.marker_width);
        linear_input_gradient(
            self.hidden_weight.values(parameters),
            &hidden_input_gradient,
            self.marker_width,
            &mut norm_output_gradient,
        );
        let (norm_scale_gradient, norm_shift_gradient) =
            split_weight_and_bias(gradient, &self.norm_scale, &self.norm_shift);
        for (((scale_gradient, shift_gradient), output_gradient), normalized_value) in
            norm_scale_gradient
                .iter_mut()
                .zip(norm_shift_gradient.iter_mut())
                .zip(&norm_output_gradient)
                .zip(&scorer_pass.normalized)
        {
            *scale_gradient += output_gradient * normalized_value;
            *shift_gradient += output_gradient;
        }
    }
}

/// `(x − mean) / √(variance + ε)`, with the biased variance LayerNorm uses,
/// accumulated in 64-bit.
fn normalized_vector(marker_state: &[f32]) -> Vec<f32> {
    let width = marker_state.len().max(1) as f64;
    let mut total = 0.0_f64;
    for marker_value in marker_state {
        total += f64::from(*marker_value);
    }
    let mean = total / width;
    let mut squared_total = 0.0_f64;
    for marker_value in marker_state {
        let centered = f64::from(*marker_value) - mean;
        squared_total += centered * centered;
    }
    let inverse_deviation = 1.0 / (squared_total / width + SCORER_LAYER_NORM_EPSILON).sqrt();
    let mut normalized = Vec::with_capacity(marker_state.len());
    for marker_value in marker_state {
        normalized.push(((f64::from(*marker_value) - mean) * inverse_deviation) as f32);
    }
    normalized
}

fn laya_marker_states(head_inputs: &HeadInputs) -> Option<Vec<&[f32]>> {
    let HeadInputs::Laya { marker_states } = head_inputs else {
        return None;
    };
    let mut marker_slices = Vec::with_capacity(marker_states.len());
    for marker_state in marker_states {
        marker_slices.push(marker_state.0.as_slice());
    }
    Some(marker_slices)
}

impl TrainableHead for LayaTrainableHead {
    fn layout(&self) -> &ParameterLayout {
        &self.layout
    }

    fn check_inputs(&self, head_inputs: &HeadInputs) -> Result<usize, HeadInputsMismatch> {
        let Some(marker_states) = laya_marker_states(head_inputs) else {
            return Err(HeadInputsMismatch::WrongFamily {
                expected: "Laya",
                found: input_family_name(head_inputs),
            });
        };
        if marker_states.is_empty() {
            return Err(HeadInputsMismatch::NoOptions);
        }
        for marker_state in &marker_states {
            if marker_state.len() != self.marker_width {
                return Err(HeadInputsMismatch::WrongWidth {
                    expected: self.marker_width,
                    found: marker_state.len(),
                });
            }
        }
        Ok(marker_states.len())
    }

    fn option_logits(&self, parameters: &[f32], head_inputs: &HeadInputs) -> Vec<f32> {
        let Some(marker_states) = laya_marker_states(head_inputs) else {
            return Vec::new();
        };
        let mut option_logits = Vec::with_capacity(marker_states.len());
        for marker_state in marker_states {
            option_logits.push(self.scorer_pass(parameters, marker_state).logit);
        }
        option_logits
    }

    fn add_gradient(
        &self,
        parameters: &[f32],
        head_inputs: &HeadInputs,
        logit_gradients: &[f32],
        gradient: &mut [f32],
    ) {
        let Some(marker_states) = laya_marker_states(head_inputs) else {
            return;
        };
        for (marker_state, logit_gradient) in marker_states.into_iter().zip(logit_gradients) {
            if *logit_gradient == 0.0 {
                continue;
            }
            let scorer_pass = self.scorer_pass(parameters, marker_state);
            self.add_option_gradient(parameters, &scorer_pass, *logit_gradient, gradient);
        }
    }
}

#[cfg(test)]
#[path = "laya_trainable_tests.rs"]
mod tests;
