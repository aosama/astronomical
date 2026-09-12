//! Forward, backward, and SGD primitives for one predictor head.

use super::network::{ExpertRoutePredictorConfig, LayerHead, LayerHeadGradients};

/// Deterministic tiny LCG; predictor weights need reproducibility, not crypto.
pub(super) fn small_random_weights(element_count: usize, seed: u64) -> Vec<f32> {
    let mut state = seed | 1;
    (0..element_count)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            // Scale to [-0.05, 0.05): small init keeps early logits near zero.
            let unit = ((state >> 33) as f32) / (u32::MAX as f32);
            (unit - 0.5) * 0.1
        })
        .collect()
}

/// Builds `[embedding; previous-route bitmap]` with an all-zero bitmap
/// whenever this layer has no observed previous route.
pub(super) fn head_input_vector(
    config: &ExpertRoutePredictorConfig,
    embedding_row: &[f32],
    previous_layer_route: Option<&Vec<u16>>,
) -> Vec<f32> {
    let mut head_input = Vec::with_capacity(config.head_input_dim());
    head_input.extend_from_slice(embedding_row);
    head_input.resize(config.head_input_dim(), 0.0);
    if let Some(previous_layer_route) = previous_layer_route {
        for routed_expert_id in previous_layer_route.iter().copied() {
            let bitmap_slot = usize::from(routed_expert_id) + config.embedding_dim;
            if let Some(bitmap_entry) = head_input.get_mut(bitmap_slot) {
                *bitmap_entry = 1.0;
            }
        }
    }
    head_input
}

/// ReLU forward through one head. Returns the expert logits.
pub(super) fn forward_head(
    config: &ExpertRoutePredictorConfig,
    head: &LayerHead,
    head_input: &[f32],
) -> Vec<f32> {
    let hidden = hidden_activations(config, head, head_input);
    let mut logits = head.output_bias.clone();
    for (logit_slot, bias) in logits.iter_mut().enumerate() {
        let weight_row_offset = logit_slot * config.hidden_dim;
        *bias += head.output_weights[weight_row_offset..weight_row_offset + config.hidden_dim]
            .iter()
            .zip(hidden.iter())
            .map(|(weight, activation)| weight * activation)
            .sum::<f32>();
    }
    logits
}

/// ReLU activations of the hidden layer for one head input.
pub(super) fn hidden_activations(
    config: &ExpertRoutePredictorConfig,
    head: &LayerHead,
    head_input: &[f32],
) -> Vec<f32> {
    head.input_bias
        .iter()
        .enumerate()
        .map(|(hidden_slot, bias)| {
            let weight_row_offset = hidden_slot * config.head_input_dim();
            let dot_product: f32 = head.input_weights
                [weight_row_offset..weight_row_offset + config.head_input_dim()]
                .iter()
                .zip(head_input.iter())
                .map(|(weight, input)| weight * input)
                .sum();
            (bias + dot_product).max(0.01 * (bias + dot_product))
        })
        .collect()
}

/// Sigmoid binary cross-entropy over all experts, plus every weight gradient.
///
/// The gradient chain is `dL/dlogit = sigmoid(logit) - target`, then standard
/// backprop through the two weight matrices, zeroing hidden contributions
/// whose ReLU was closed. Routed ids must be sorted ascending, matching the
/// observation history's compacted form.
pub(super) fn backward_head(
    config: &ExpertRoutePredictorConfig,
    head: &LayerHead,
    head_input: &[f32],
    routed_expert_ids: &[u16],
) -> (f32, LayerHeadGradients) {
    let logits = forward_head(config, head, head_input);
    let hidden = hidden_activations(config, head, head_input);
    let mut logit_gradients = vec![0.0_f32; config.expert_count];
    let mut loss = 0.0_f32;
    let mut routed_expert_id_iter = routed_expert_ids.iter().copied().peekable();
    for (logit_slot, logit) in logits.iter().enumerate() {
        let is_routed = routed_expert_id_iter
            .peek()
            .is_some_and(|next_routed_expert_id| usize::from(*next_routed_expert_id) == logit_slot);
        if is_routed {
            routed_expert_id_iter.next();
        }
        let target = if is_routed { 1.0 } else { 0.0 };
        let probability = sigmoid(*logit);
        loss -= if target == 1.0 {
            probability.clamp(1.0e-7, 1.0).ln()
        } else {
            (1.0 - probability).clamp(1.0e-7, 1.0).ln()
        };
        logit_gradients[logit_slot] = probability - target;
    }
    let mut gradients = LayerHeadGradients::zeroed(config);
    // Output weights and bias: dL/dW2[k][h] = dL/dlogit[k] * hidden[h].
    for (logit_slot, logit_gradient) in logit_gradients.iter().enumerate() {
        gradients.output_bias[logit_slot] = *logit_gradient;
        if *logit_gradient == 0.0 {
            continue;
        }
        let weight_row_offset = logit_slot * config.hidden_dim;
        for (hidden_slot, gradient) in gradients.output_weights
            [weight_row_offset..weight_row_offset + config.hidden_dim]
            .iter_mut()
            .enumerate()
        {
            *gradient = logit_gradient * hidden[hidden_slot];
        }
    }
    // Hidden deltas: backprop through W2 with the ReLU mask.
    let mut hidden_gradients = vec![0.0_f32; config.hidden_dim];
    for (logit_slot, logit_gradient) in logit_gradients.iter().enumerate() {
        if *logit_gradient == 0.0 {
            continue;
        }
        let weight_row_offset = logit_slot * config.hidden_dim;
        for (hidden_slot, weight) in head.output_weights
            [weight_row_offset..weight_row_offset + config.hidden_dim]
            .iter()
            .enumerate()
        {
            hidden_gradients[hidden_slot] += logit_gradient * weight;
        }
    }
    // Input weights, bias, and the returned input gradient for the embedding.
    // Leaky ReLU keeps a 0.01 slope below zero, so closed units still train.
    for (hidden_slot, hidden_gradient) in hidden_gradients.iter().enumerate() {
        let leak_factor = if hidden[hidden_slot] > 0.0 { 1.0 } else { 0.01 };
        let gated_gradient = *hidden_gradient * leak_factor;
        gradients.input_bias[hidden_slot] = gated_gradient;
        if gated_gradient == 0.0 {
            continue;
        }
        let weight_row_offset = hidden_slot * config.head_input_dim();
        for (input_slot, gradient) in gradients.input_weights
            [weight_row_offset..weight_row_offset + config.head_input_dim()]
            .iter_mut()
            .enumerate()
        {
            *gradient = gated_gradient * head_input[input_slot];
        }
    }
    // dL/dhead_input = W1^T · hidden_delta, leaky units scaled by the slope.
    let mut input_gradient = vec![0.0_f32; config.head_input_dim()];
    for (hidden_slot, hidden_gradient) in hidden_gradients.iter().enumerate() {
        let leak_factor = if hidden[hidden_slot] > 0.0 { 1.0 } else { 0.01 };
        let gated_gradient = *hidden_gradient * leak_factor;
        if gated_gradient == 0.0 {
            continue;
        }
        let weight_row_offset = hidden_slot * config.head_input_dim();
        for (input_slot, weight) in head.input_weights
            [weight_row_offset..weight_row_offset + config.head_input_dim()]
            .iter()
            .enumerate()
        {
            input_gradient[input_slot] += gated_gradient * weight;
        }
    }
    gradients.input_gradient = input_gradient;
    (loss, gradients)
}

/// Applies one in-place SGD step to a head.
pub(super) fn apply_head_sgd_step(
    config: &ExpertRoutePredictorConfig,
    head: &mut LayerHead,
    gradients: &LayerHeadGradients,
    learning_rate: f32,
) {
    for (weight, gradient) in head
        .input_weights
        .iter_mut()
        .zip(gradients.input_weights.iter())
    {
        *weight -= learning_rate * gradient;
    }
    for (bias, gradient) in head.input_bias.iter_mut().zip(gradients.input_bias.iter()) {
        *bias -= learning_rate * gradient;
    }
    for (weight, gradient) in head
        .output_weights
        .iter_mut()
        .zip(gradients.output_weights.iter())
    {
        *weight -= learning_rate * gradient;
    }
    for (bias, gradient) in head
        .output_bias
        .iter_mut()
        .zip(gradients.output_bias.iter())
    {
        *bias -= learning_rate * gradient;
    }
    let _ = config;
}

pub(super) fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}
