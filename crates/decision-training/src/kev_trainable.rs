//! Kev's pointer head as a trainable part: `z_k = (K h_k + b_K) · (Q h_d + b_Q) / √P`.
//!
//! The decision token's state `h_d` and each option's closing state `h_k`
//! come from the frozen backbone, so the whole head trains: both projections
//! and both biases, about half a million numbers for Kev-0.8B.

use astronomical_decision_contract::HeadInputs;

use crate::dense_math::{dot, linear_forward, linear_parameter_gradients};
use crate::head_parameters::{ParameterLayout, ParameterTensor, split_weight_and_bias};
use crate::trainable_head::{HeadInputsMismatch, TrainableHead, input_family_name};

/// Kev's head for one pointer dimension and backbone width.
pub struct KevTrainableHead {
    layout: ParameterLayout,
    pointer_dimension: usize,
    hidden_size: usize,
    query_weight: ParameterTensor,
    query_bias: ParameterTensor,
    key_weight: ParameterTensor,
    key_bias: ParameterTensor,
}

impl KevTrainableHead {
    #[must_use]
    pub fn new(pointer_dimension: usize, hidden_size: usize) -> Self {
        let layout = ParameterLayout::new(&[
            ("q.weight", vec![pointer_dimension, hidden_size]),
            ("q.bias", vec![pointer_dimension]),
            ("k.weight", vec![pointer_dimension, hidden_size]),
            ("k.bias", vec![pointer_dimension]),
        ]);
        Self {
            query_weight: layout.tensor("q.weight").clone(),
            query_bias: layout.tensor("q.bias").clone(),
            key_weight: layout.tensor("k.weight").clone(),
            key_bias: layout.tensor("k.bias").clone(),
            layout,
            pointer_dimension,
            hidden_size,
        }
    }

    fn scale(&self) -> f32 {
        1.0 / (self.pointer_dimension as f32).sqrt()
    }

    /// `Q h_d + b_Q` and each `K h_k + b_K`.
    fn projections(
        &self,
        parameters: &[f32],
        decision_state: &[f32],
        option_states: &[&[f32]],
    ) -> (Vec<f32>, Vec<Vec<f32>>) {
        let mut query_vector = Vec::with_capacity(self.pointer_dimension);
        linear_forward(
            self.query_weight.values(parameters),
            self.query_bias.values(parameters),
            decision_state,
            &mut query_vector,
        );
        let mut key_vectors = Vec::with_capacity(option_states.len());
        for option_state in option_states {
            let mut key_vector = Vec::with_capacity(self.pointer_dimension);
            linear_forward(
                self.key_weight.values(parameters),
                self.key_bias.values(parameters),
                option_state,
                &mut key_vector,
            );
            key_vectors.push(key_vector);
        }
        (query_vector, key_vectors)
    }
}

/// The decision state and option states of Kev inputs, or `None` for another family.
fn kev_states(head_inputs: &HeadInputs) -> Option<(&[f32], Vec<&[f32]>)> {
    let HeadInputs::Kev {
        decision_state,
        option_states,
    } = head_inputs
    else {
        return None;
    };
    let mut option_slices = Vec::with_capacity(option_states.len());
    for option_state in option_states {
        option_slices.push(option_state.0.as_slice());
    }
    Some((decision_state.0.as_slice(), option_slices))
}

impl TrainableHead for KevTrainableHead {
    fn layout(&self) -> &ParameterLayout {
        &self.layout
    }

    fn check_inputs(&self, head_inputs: &HeadInputs) -> Result<usize, HeadInputsMismatch> {
        let Some((decision_state, option_states)) = kev_states(head_inputs) else {
            return Err(HeadInputsMismatch::WrongFamily {
                expected: "Kev",
                found: input_family_name(head_inputs),
            });
        };
        if option_states.is_empty() {
            return Err(HeadInputsMismatch::NoOptions);
        }
        for state in std::iter::once(decision_state).chain(option_states.iter().copied()) {
            if state.len() != self.hidden_size {
                return Err(HeadInputsMismatch::WrongWidth {
                    expected: self.hidden_size,
                    found: state.len(),
                });
            }
        }
        Ok(option_states.len())
    }

    fn option_logits(&self, parameters: &[f32], head_inputs: &HeadInputs) -> Vec<f32> {
        let Some((decision_state, option_states)) = kev_states(head_inputs) else {
            return Vec::new();
        };
        let (query_vector, key_vectors) =
            self.projections(parameters, decision_state, &option_states);
        let scale = self.scale();
        let mut option_logits = Vec::with_capacity(key_vectors.len());
        for key_vector in &key_vectors {
            option_logits.push(dot(key_vector, &query_vector) * scale);
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
        let Some((decision_state, option_states)) = kev_states(head_inputs) else {
            return;
        };
        let (query_vector, key_vectors) =
            self.projections(parameters, decision_state, &option_states);
        let scale = self.scale();
        // dz_k/dq = s·k_k and dz_k/dk_k = s·q.
        let mut query_gradient = vec![0.0_f32; self.pointer_dimension];
        let mut key_gradient = Vec::with_capacity(self.pointer_dimension);
        let (key_weight_gradient, key_bias_gradient) =
            split_weight_and_bias(gradient, &self.key_weight, &self.key_bias);
        for ((key_vector, option_state), logit_gradient) in
            key_vectors.iter().zip(&option_states).zip(logit_gradients)
        {
            let scaled_gradient = logit_gradient * scale;
            for (query_gradient_value, key_value) in query_gradient.iter_mut().zip(key_vector) {
                *query_gradient_value += scaled_gradient * key_value;
            }
            key_gradient.clear();
            for query_value in &query_vector {
                key_gradient.push(scaled_gradient * query_value);
            }
            linear_parameter_gradients(
                option_state,
                &key_gradient,
                key_weight_gradient,
                key_bias_gradient,
            );
        }
        let (query_weight_gradient, query_bias_gradient) =
            split_weight_and_bias(gradient, &self.query_weight, &self.query_bias);
        linear_parameter_gradients(
            decision_state,
            &query_gradient,
            query_weight_gradient,
            query_bias_gradient,
        );
    }
}

#[cfg(test)]
#[path = "kev_trainable_tests.rs"]
mod tests;
