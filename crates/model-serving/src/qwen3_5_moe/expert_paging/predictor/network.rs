//! Predictor network: weights, forward pass, backward pass, SGD update.
//!
//! Architecture. One shared token-embedding table feeds every sparse decoder
//! layer. Each layer owns a two-layer multilayer perceptron whose input is the
//! embedding row concatenated with that layer's previous-route bitmap, so a
//! head only ever sees the routing history of its own layer. The output is one
//! logit per expert; the training loss is a sigmoid binary cross-entropy over
//! all experts with the true routed experts as positives.
//!
//! Why dense bitmaps and plain loops. The network is tiny (a few million
//! `f32` parameters), one sample at a time, on the host CPU. Every alternative
//! considered — sparse embeddings of the bitmap, a dependency, SIMD intrinsics
//! — either added state this issue forbids or optimization this issue must
//! measure before adopting. Plain loops keep the backward pass provable
//! against a brute-force numerical reference.

use super::ops::{apply_head_sgd_step, forward_head, head_input_vector, small_random_weights};
use crate::qwen3_5_moe::expert_paging::route_observation::RouteObservationRecord;

/// Fixed geometry and hyperparameters for one predictor.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExpertRoutePredictorConfig {
    pub layer_count: usize,
    pub expert_count: usize,
    pub vocabulary_size: u32,
    pub embedding_dim: usize,
    pub hidden_dim: usize,
    /// SGD step size for every weight tensor.
    pub learning_rate: f32,
    /// Deterministic weight initialization seed; no external randomness.
    pub seed: u64,
}

impl ExpertRoutePredictorConfig {
    /// Input width of one layer head: embedding row plus previous-route bitmap.
    #[must_use]
    pub const fn head_input_dim(&self) -> usize {
        self.embedding_dim + self.expert_count
    }
}

/// One layer head's weights.
#[derive(Clone, Debug)]
pub(super) struct LayerHead {
    /// Row-major `[hidden_dim][head_input_dim]`.
    pub(super) input_weights: Vec<f32>,
    pub(super) input_bias: Vec<f32>,
    /// Row-major `[expert_count][hidden_dim]`.
    pub(super) output_weights: Vec<f32>,
    pub(super) output_bias: Vec<f32>,
}

/// Gradients for one head from one labeled example.
#[derive(Debug, Default)]
pub(super) struct LayerHeadGradients {
    pub(super) input_weights: Vec<f32>,
    pub(super) input_bias: Vec<f32>,
    pub(super) output_weights: Vec<f32>,
    pub(super) output_bias: Vec<f32>,
    /// Gradient wrt the head input vector, needed by the shared embedding.
    pub(super) input_gradient: Vec<f32>,
}

impl LayerHeadGradients {
    pub(super) fn zeroed(config: &ExpertRoutePredictorConfig) -> Self {
        Self {
            input_weights: vec![0.0; config.hidden_dim * config.head_input_dim()],
            input_bias: vec![0.0; config.hidden_dim],
            output_weights: vec![0.0; config.expert_count * config.hidden_dim],
            output_bias: vec![0.0; config.expert_count],
            input_gradient: vec![0.0; config.head_input_dim()],
        }
    }
}

/// The complete predictor: shared embedding plus one head per sparse layer.
#[derive(Clone, Debug)]
pub struct ExpertRoutePredictor {
    pub(super) config: ExpertRoutePredictorConfig,
    /// Row-major `[vocabulary_size][embedding_dim]`.
    pub(super) embedding_table: Vec<f32>,
    pub(super) layer_heads: Vec<LayerHead>,
}

impl ExpertRoutePredictor {
    #[must_use]
    pub fn new(config: ExpertRoutePredictorConfig) -> Self {
        let layer_heads: Vec<LayerHead> = (0..config.layer_count)
            .map(|layer_index| {
                let layer_seed = config.seed
                    ^ u64::from(u32::try_from(layer_index).unwrap_or(0))
                        .wrapping_mul(0x9E37_79B9_7F4A_7C15);
                LayerHead {
                    input_weights: small_random_weights(
                        config.hidden_dim * config.head_input_dim(),
                        layer_seed,
                    ),
                    input_bias: vec![0.0; config.hidden_dim],
                    output_weights: small_random_weights(
                        config.expert_count * config.hidden_dim,
                        layer_seed.rotate_left(17),
                    ),
                    output_bias: vec![0.0; config.expert_count],
                }
            })
            .collect();
        Self {
            embedding_table: small_random_weights(
                usize::try_from(config.vocabulary_size).unwrap_or(0) * config.embedding_dim,
                config.seed,
            )
            .into_iter()
            // Embeddings are looked up, not summed, so they need a larger
            // init than the dense weights or the per-token signal starves
            // behind the shared bias.
            .map(|weight| weight * 10.0)
            .collect(),
            config,
            layer_heads,
        }
    }

    #[must_use]
    pub fn config(&self) -> &ExpertRoutePredictorConfig {
        &self.config
    }

    /// Frozen 1x1-convolution copy of every layer head for Core ML export.
    #[cfg(target_os = "macos")]
    #[must_use]
    pub fn convolution_snapshot(
        &self,
    ) -> astronomical_runtime_integration::PredictorAneConvolutionSnapshot {
        let mut conv1_weights = Vec::new();
        let mut conv1_bias = Vec::new();
        let mut conv2_weights = Vec::new();
        let mut conv2_bias = Vec::new();
        for layer_head in &self.layer_heads {
            conv1_weights.extend_from_slice(&layer_head.input_weights);
            conv1_bias.extend_from_slice(&layer_head.input_bias);
            conv2_weights.extend_from_slice(&layer_head.output_weights);
            conv2_bias.extend_from_slice(&layer_head.output_bias);
        }
        astronomical_runtime_integration::PredictorAneConvolutionSnapshot {
            layer_count: self.config.layer_count,
            expert_count: self.config.expert_count,
            input_dim: self.config.head_input_dim(),
            hidden_dim: self.config.hidden_dim,
            conv1_weights,
            conv1_bias,
            conv2_weights,
            conv2_bias,
        }
    }

    /// Total trainable parameters, for memory reporting.
    #[must_use]
    pub fn parameter_count(&self) -> u64 {
        let head_parameter_count = self
            .config
            .hidden_dim
            .saturating_mul(self.config.head_input_dim())
            .saturating_add(self.config.hidden_dim)
            .saturating_add(
                self.config
                    .expert_count
                    .saturating_mul(self.config.hidden_dim),
            )
            .saturating_add(self.config.expert_count);
        u64::try_from(self.embedding_table.len()).unwrap_or(u64::MAX)
            + u64::try_from(head_parameter_count.saturating_mul(self.config.layer_count))
                .unwrap_or(u64::MAX)
    }

    /// Per-layer expert logits for one token. Layers whose previous-route
    /// bitmap is absent (first observed token, or an unobserved layer) receive
    /// an all-zero bitmap, which is the honest "no history" input.
    #[must_use]
    pub fn forward_logits(
        &self,
        token_id: u32,
        previous_token_route: Option<&[Option<Vec<u16>>]>,
    ) -> Vec<Vec<f32>> {
        let embedding_row = self.embedding_row(token_id);
        self.layer_heads
            .iter()
            .enumerate()
            .map(|(layer_index, head)| {
                let head_input = head_input_vector(
                    &self.config,
                    embedding_row,
                    previous_token_route
                        .and_then(|route| route.get(layer_index))
                        .and_then(|maybe_layer_route| maybe_layer_route.as_ref()),
                );
                forward_head(&self.config, head, &head_input)
            })
            .collect()
    }

    /// Layer-major `[layer][head_input_dim]` activations for a Core ML snapshot.
    #[must_use]
    pub fn packed_head_inputs(
        &self,
        token_id: u32,
        previous_token_route: Option<&[Option<Vec<u16>>]>,
    ) -> Vec<f32> {
        let embedding_row = self.embedding_row(token_id);
        let mut packed_head_inputs =
            Vec::with_capacity(self.config.layer_count * self.config.head_input_dim());
        for layer_index in 0..self.config.layer_count {
            packed_head_inputs.extend(head_input_vector(
                &self.config,
                embedding_row,
                previous_token_route
                    .and_then(|route| route.get(layer_index))
                    .and_then(|maybe_layer_route| maybe_layer_route.as_ref()),
            ));
        }
        packed_head_inputs
    }

    /// One labeled example: forward, backward, and one SGD step. Returns the
    /// summed loss over the layers that had labels. Unlabeled layers
    /// (`None` routes) contribute no gradient, so an unobserved layer never
    /// corrupts its head.
    pub fn train_on_record(&mut self, record: &RouteObservationRecord) -> f32 {
        let (total_loss, gradients_per_layer, embedding_gradient) = self.record_gradients(record);
        let learning_rate = self.config.learning_rate;
        for (layer_index, gradients) in gradients_per_layer.into_iter().enumerate() {
            let Some(gradients) = gradients else {
                continue;
            };
            if let Some(head) = self.layer_heads.get_mut(layer_index) {
                apply_head_sgd_step(&self.config, head, &gradients, learning_rate);
            }
        }
        let embedding_row_offset = usize::try_from(record.input_token_id).unwrap_or(usize::MAX)
            * self.config.embedding_dim;
        for (weight, gradient) in self.embedding_table
            [embedding_row_offset..embedding_row_offset + self.config.embedding_dim]
            .iter_mut()
            .zip(embedding_gradient.iter())
        {
            *weight -= learning_rate * gradient;
        }
        total_loss
    }

    pub(super) fn embedding_row(&self, token_id: u32) -> &[f32] {
        let row_offset =
            usize::try_from(token_id).unwrap_or(usize::MAX) * self.config.embedding_dim;
        &self.embedding_table[row_offset..row_offset + self.config.embedding_dim]
    }
}
