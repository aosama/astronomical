//! Kev's pointer head, computed in float32 on the host as the reference
//! computes it: `z_k = (K h_k + b_K) · (Q h_d + b_Q) / √P`, where `h_d` is the
//! decision token's hidden state, `h_k` option k's closing token's, and `P`
//! the pointer dimension.
//!
//! The head is two small projections per question, about a million
//! multiply-adds at Kev-9B's width; doing it on the host keeps it in the
//! reference's precision without a graph per question.

use std::path::Path;

use astronomical_decision_contract::DecisionFailure;
use safetensors::{Dtype, SafeTensors};

pub(crate) const KEV_HEAD_FILE_NAME: &str = "kev_head.safetensors";
/// The head is two 256-wide projections; anything near this size is not one.
const MAXIMUM_HEAD_FILE_BYTES: u64 = 256 * 1024 * 1024;

/// One linear projection: `weight` is `[pointer_dimension, hidden_size]`.
#[derive(Clone, Debug, PartialEq)]
struct Projection {
    weight: Vec<f32>,
    bias: Vec<f32>,
}

/// The loaded head.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct KevPointerHead {
    query: Projection,
    key: Projection,
    pointer_dimension: usize,
    hidden_size: usize,
}

impl KevPointerHead {
    pub(crate) fn read(
        model_directory: &Path,
        pointer_dimension: usize,
    ) -> Result<Self, DecisionFailure> {
        let head_path = model_directory.join(KEV_HEAD_FILE_NAME);
        let head_size = std::fs::metadata(&head_path)
            .map_err(|_| unreadable_head("it is missing"))?
            .len();
        if head_size > MAXIMUM_HEAD_FILE_BYTES {
            return Err(unreadable_head("it is too large"));
        }
        let head_bytes =
            std::fs::read(&head_path).map_err(|_| unreadable_head("it cannot be read"))?;
        Self::from_safetensors_bytes(&head_bytes, pointer_dimension)
    }

    pub(crate) fn from_safetensors_bytes(
        head_bytes: &[u8],
        pointer_dimension: usize,
    ) -> Result<Self, DecisionFailure> {
        let head_tensors = SafeTensors::deserialize(head_bytes)
            .map_err(|_| unreadable_head("it is not safetensors"))?;
        let query = projection(&head_tensors, "q", pointer_dimension)?;
        let key = projection(&head_tensors, "k", pointer_dimension)?;
        let hidden_size = query.weight.len() / pointer_dimension;
        if key.weight.len() != query.weight.len() || hidden_size == 0 {
            return Err(unreadable_head(
                "its projections do not have matching shapes",
            ));
        }
        Ok(Self {
            query,
            key,
            pointer_dimension,
            hidden_size,
        })
    }

    pub(crate) const fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// One logit per option, before the temperature, for one question.
    pub(crate) fn option_logits(
        &self,
        decision_hidden_state: &[f32],
        option_hidden_states: &[&[f32]],
    ) -> Vec<f32> {
        let query_vector = project(&self.query, decision_hidden_state, self.pointer_dimension);
        let scale = 1.0 / (self.pointer_dimension as f32).sqrt();
        let mut option_logits = Vec::with_capacity(option_hidden_states.len());
        for option_hidden_state in option_hidden_states {
            let key_vector = project(&self.key, option_hidden_state, self.pointer_dimension);
            let mut dot_product = 0.0_f32;
            for (key_value, query_value) in key_vector.iter().zip(&query_vector) {
                dot_product += key_value * query_value;
            }
            option_logits.push(dot_product * scale);
        }
        option_logits
    }
}

/// `weight · hidden + bias`, row by row.
fn project(projection: &Projection, hidden_state: &[f32], pointer_dimension: usize) -> Vec<f32> {
    let hidden_size = hidden_state.len();
    let mut projected = Vec::with_capacity(pointer_dimension);
    for (output_index, bias_value) in projection.bias.iter().enumerate() {
        let weight_row =
            &projection.weight[output_index * hidden_size..(output_index + 1) * hidden_size];
        let mut accumulated = 0.0_f32;
        for (weight_value, hidden_value) in weight_row.iter().zip(hidden_state) {
            accumulated += weight_value * hidden_value;
        }
        projected.push(accumulated + bias_value);
    }
    projected
}

/// The probabilities Kev reports: `softmax(logits / temperature)`. Unlike
/// the shared calibration, the temperature is used as the checkpoint gives
/// it; the reference never clamps it.
pub(crate) fn kev_probabilities(option_logits: &[f32], temperature: f64) -> Vec<f64> {
    let mut largest_scaled_logit = f64::NEG_INFINITY;
    for option_logit in option_logits {
        largest_scaled_logit = largest_scaled_logit.max(f64::from(*option_logit) / temperature);
    }
    let mut weights = Vec::with_capacity(option_logits.len());
    let mut weight_total = 0.0_f64;
    for option_logit in option_logits {
        let weight = (f64::from(*option_logit) / temperature - largest_scaled_logit).exp();
        weight_total += weight;
        weights.push(weight);
    }
    for weight in &mut weights {
        *weight /= weight_total;
    }
    weights
}

fn projection(
    head_tensors: &SafeTensors<'_>,
    projection_name: &str,
    pointer_dimension: usize,
) -> Result<Projection, DecisionFailure> {
    let weight = float32_tensor(head_tensors, &format!("{projection_name}.weight"))?;
    let bias = float32_tensor(head_tensors, &format!("{projection_name}.bias"))?;
    if bias.len() != pointer_dimension || weight.len() % pointer_dimension != 0 {
        return Err(unreadable_head(
            "a projection's width is not the pointer dimension",
        ));
    }
    Ok(Projection { weight, bias })
}

fn float32_tensor(
    head_tensors: &SafeTensors<'_>,
    tensor_name: &str,
) -> Result<Vec<f32>, DecisionFailure> {
    let tensor_view = head_tensors
        .tensor(tensor_name)
        .map_err(|_| unreadable_head("a projection tensor is missing"))?;
    if tensor_view.dtype() != Dtype::F32 {
        return Err(unreadable_head("its tensors are not float32"));
    }
    let (whole_values, _) = tensor_view.data().as_chunks::<4>();
    let mut values = Vec::with_capacity(whole_values.len());
    for value_bytes in whole_values {
        values.push(f32::from_le_bytes(*value_bytes));
    }
    Ok(values)
}

fn unreadable_head(reason: &str) -> DecisionFailure {
    DecisionFailure::fatal_execution(&format!("{KEV_HEAD_FILE_NAME} cannot be used: {reason}"))
}

#[cfg(test)]
#[path = "pointer_head_tests.rs"]
mod tests;
