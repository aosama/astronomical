//! The layers Clef's joint head is built from, as MLX graphs over float32
//! `[1, length, width]` activations: PyTorch's `Linear`, `LayerNorm`,
//! `MultiheadAttention` (batch first, packed input projection, no mask), the
//! reference's `EvidenceRoutingLayer`, and `TransformerDecoderLayer` with
//! `norm_first=True` and exact GELU.

use astronomical_decision_contract::DecisionFailure;
use astronomical_runtime_integration::MlxRuntime;
use mlx_bridge::{MlxArray, MlxCError};

use super::head_weights::ClefHeadWeights;

/// PyTorch's `LayerNorm` default, which the head was trained with.
const HEAD_LAYER_NORM_EPSILON: f32 = 1e-5;

/// The runtime, the weights and the head count every layer needs.
pub(crate) struct ClefHeadLayers<'head> {
    pub(crate) runtime: &'head MlxRuntime,
    pub(crate) weights: &'head ClefHeadWeights,
    pub(crate) attention_head_count: usize,
}

impl ClefHeadLayers<'_> {
    /// `input @ weight.T (+ bias)`.
    pub(crate) fn linear(
        &self,
        linear_prefix: &str,
        input: &MlxArray,
        has_bias: bool,
    ) -> Result<MlxArray, DecisionFailure> {
        let weight = self.weights.tensor(&format!("{linear_prefix}.weight"))?;
        let projected = self
            .runtime
            .matmul(input, &self.runtime.transpose(weight).map_err(head_math)?)
            .map_err(head_math)?;
        if !has_bias {
            return Ok(projected);
        }
        let bias = self.weights.tensor(&format!("{linear_prefix}.bias"))?;
        self.runtime.add(&projected, bias).map_err(head_math)
    }

    pub(crate) fn layer_norm(
        &self,
        norm_prefix: &str,
        input: &MlxArray,
    ) -> Result<MlxArray, DecisionFailure> {
        self.runtime
            .layer_norm(
                input,
                self.weights.tensor(&format!("{norm_prefix}.weight"))?,
                self.weights.tensor(&format!("{norm_prefix}.bias"))?,
                HEAD_LAYER_NORM_EPSILON,
            )
            .map_err(head_math)
    }

    /// `torch.nn.MultiheadAttention(batch_first=True)` without a mask:
    /// `queries` is `[1, Lq, W]`, `keys_and_values` `[1, Lk, W]`.
    pub(crate) fn multihead_attention(
        &self,
        attention_prefix: &str,
        queries: &MlxArray,
        keys_and_values: &MlxArray,
    ) -> Result<MlxArray, DecisionFailure> {
        let projected_queries = self.packed_projection(attention_prefix, "query", queries)?;
        let projected_keys = self.packed_projection(attention_prefix, "key", keys_and_values)?;
        let projected_values =
            self.packed_projection(attention_prefix, "value", keys_and_values)?;
        let width = queries.shape()[2];
        let head_width = width / self.attention_head_count as i32;
        let attended = self
            .runtime
            .scaled_dot_product_attention(
                &self.split_heads(&projected_queries)?,
                &self.split_heads(&projected_keys)?,
                &self.split_heads(&projected_values)?,
                (head_width as f32).powf(-0.5),
            )
            .map_err(head_math)?;
        let merged_heads = self
            .runtime
            .transpose_axes(&attended, &[0, 2, 1, 3])
            .map_err(head_math)?;
        let query_length = queries.shape()[1];
        let joined = self
            .runtime
            .reshape(&merged_heads, &[1, query_length, width])
            .map_err(head_math)?;
        self.linear(&format!("{attention_prefix}.out_proj"), &joined, true)
    }

    /// `EvidenceRoutingLayer`: queries attend to the normalized memory, then
    /// a pre-norm GELU feed-forward; both with residuals.
    pub(crate) fn evidence_routing_layer(
        &self,
        layer_prefix: &str,
        queries: &MlxArray,
        memory: &MlxArray,
    ) -> Result<MlxArray, DecisionFailure> {
        let normalized_memory = self.layer_norm(&format!("{layer_prefix}.memory_norm"), memory)?;
        let normalized_queries = self.layer_norm(&format!("{layer_prefix}.query_norm"), queries)?;
        let routed = self.multihead_attention(
            &format!("{layer_prefix}.attention"),
            &normalized_queries,
            &normalized_memory,
        )?;
        let queries = self.runtime.add(queries, &routed).map_err(head_math)?;
        let feedforward_input =
            self.layer_norm(&format!("{layer_prefix}.feedforward_norm"), &queries)?;
        let feedforward_output = self.feedforward(
            &format!("{layer_prefix}.feedforward.0"),
            &format!("{layer_prefix}.feedforward.3"),
            &feedforward_input,
        )?;
        self.runtime
            .add(&queries, &feedforward_output)
            .map_err(head_math)
    }

    /// `TransformerDecoderLayer(norm_first=True, activation="gelu")`: self
    /// attention, attention to the unnormalized memory, feed-forward.
    pub(crate) fn decoder_layer(
        &self,
        layer_prefix: &str,
        fields: &MlxArray,
        memory: &MlxArray,
    ) -> Result<MlxArray, DecisionFailure> {
        let self_attention_input = self.layer_norm(&format!("{layer_prefix}.norm1"), fields)?;
        let self_attended = self.multihead_attention(
            &format!("{layer_prefix}.self_attn"),
            &self_attention_input,
            &self_attention_input,
        )?;
        let fields = self
            .runtime
            .add(fields, &self_attended)
            .map_err(head_math)?;
        let cross_attention_input = self.layer_norm(&format!("{layer_prefix}.norm2"), &fields)?;
        let cross_attended = self.multihead_attention(
            &format!("{layer_prefix}.multihead_attn"),
            &cross_attention_input,
            memory,
        )?;
        let fields = self
            .runtime
            .add(&fields, &cross_attended)
            .map_err(head_math)?;
        let feedforward_input = self.layer_norm(&format!("{layer_prefix}.norm3"), &fields)?;
        let feedforward_output = self.feedforward(
            &format!("{layer_prefix}.linear1"),
            &format!("{layer_prefix}.linear2"),
            &feedforward_input,
        )?;
        self.runtime
            .add(&fields, &feedforward_output)
            .map_err(head_math)
    }

    /// `Linear → GELU → Linear`.
    pub(crate) fn feedforward(
        &self,
        expanding_prefix: &str,
        contracting_prefix: &str,
        input: &MlxArray,
    ) -> Result<MlxArray, DecisionFailure> {
        let expanded = self.linear(expanding_prefix, input, true)?;
        let activated = self.runtime.gelu(&expanded).map_err(head_math)?;
        self.linear(contracting_prefix, &activated, true)
    }

    fn packed_projection(
        &self,
        attention_prefix: &str,
        part_name: &str,
        input: &MlxArray,
    ) -> Result<MlxArray, DecisionFailure> {
        let weight = self
            .weights
            .tensor(&format!("{attention_prefix}.{part_name}_weight"))?;
        let bias = self
            .weights
            .tensor(&format!("{attention_prefix}.{part_name}_bias"))?;
        let projected = self
            .runtime
            .matmul(input, &self.runtime.transpose(weight).map_err(head_math)?)
            .map_err(head_math)?;
        self.runtime.add(&projected, bias).map_err(head_math)
    }

    /// `[1, L, W]` to `[1, heads, L, W / heads]`.
    fn split_heads(&self, projected: &MlxArray) -> Result<MlxArray, DecisionFailure> {
        let projected_shape = projected.shape();
        let head_count = self.attention_head_count as i32;
        let per_head = self
            .runtime
            .reshape(
                projected,
                &[
                    1,
                    projected_shape[1],
                    head_count,
                    projected_shape[2] / head_count,
                ],
            )
            .map_err(head_math)?;
        self.runtime
            .transpose_axes(&per_head, &[0, 2, 1, 3])
            .map_err(head_math)
    }
}

pub(crate) fn head_math(mlx_error: MlxCError) -> DecisionFailure {
    DecisionFailure::fatal_execution(&format!("Clef's joint head failed: {mlx_error}"))
}
