//! Decomposed stock-MLX implementation used only as a model-wiring oracle.
//!
//! This module avoids every Laguna model helper under test. It intentionally
//! uses chronological cache arrays and explicit masks rather than production's
//! physical ring and native-causal fast paths; agreement therefore establishes
//! semantics across two different graph constructions, not duplicated wiring.

use std::collections::HashMap;

use astronomical_model_serving::{
    LagunaAttentionProjection, LagunaCacheDescriptor, LagunaExpertProjection,
    LagunaFeedForwardDescriptor, LagunaGatingKind, LagunaGlobalTensorRole, LagunaLayerTensorRole,
    LagunaRopeDescriptor, LagunaTargetContract, LagunaTensorComponent, LagunaTensorId,
    PerformanceAttribution, build_causal_sliding_window_mask,
    compute_yarn_rope_frequency_denominators,
};
use astronomical_runtime_integration::{MlxArray, MlxDtype, MlxRuntime, MlxRuntimeError};

use super::moe_operations::reference_moe;

pub(super) struct ReferenceDecoderState {
    layers: Vec<ReferenceLayerState>,
}

struct ReferenceLayerState {
    // Chronological arrays are simpler than the production ring and make the
    // visible token set explicit at the cost of test-only concatenation.
    keys: Option<MlxArray>,
    values: Option<MlxArray>,
    absolute_position: i32,
    window_size: Option<i32>,
}

impl ReferenceDecoderState {
    pub(super) fn new(contract: &LagunaTargetContract) -> Self {
        let layers = contract
            .layers()
            .iter()
            .map(|layer| ReferenceLayerState {
                keys: None,
                values: None,
                absolute_position: 0,
                window_size: match layer.attention().cache() {
                    LagunaCacheDescriptor::AppendOnly => None,
                    LagunaCacheDescriptor::Rotating { window_size } => Some(*window_size as i32),
                },
            })
            .collect();
        Self { layers }
    }

    pub(super) fn absolute_position(&self, layer_index: usize) -> i32 {
        self.layers[layer_index].absolute_position
    }

    pub(super) fn committed_token_count(&self, layer_index: usize) -> i32 {
        let layer = &self.layers[layer_index];
        layer.window_size.map_or(layer.absolute_position, |window| {
            window.min(layer.absolute_position)
        })
    }
}

pub(super) fn reference_forward(
    runtime: &MlxRuntime,
    contract: &LagunaTargetContract,
    tensors: &HashMap<LagunaTensorId, MlxArray>,
    token_ids: &MlxArray,
    state: &mut ReferenceDecoderState,
) -> Result<MlxArray, MlxRuntimeError> {
    let embedding = tensor(tensors, global(LagunaGlobalTensorRole::TokenEmbedding));
    let mut hidden = runtime.take_axis(embedding, token_ids, 0)?;
    if hidden.shape().len() == 2 {
        hidden = runtime.reshape(&hidden, &[1, hidden.shape()[0], hidden.shape()[1]])?;
    }
    let epsilon = contract.model().rms_norm_epsilon() as f32;
    for layer in contract.layers() {
        let layer_index = layer.layer_index();
        let normalized = runtime.rms_norm(
            &hidden,
            tensor(
                tensors,
                layer_id(layer_index, LagunaLayerTensorRole::InputNormalization),
            ),
            epsilon,
        )?;
        let attention_delta = reference_attention(
            runtime,
            tensors,
            layer_index,
            layer.attention(),
            &normalized,
            &mut state.layers[layer_index],
            epsilon,
        )?;
        let after_attention = runtime.add(&hidden, &attention_delta)?;
        let normalized_after_attention = runtime.rms_norm(
            &after_attention,
            tensor(
                tensors,
                layer_id(
                    layer_index,
                    LagunaLayerTensorRole::PostAttentionNormalization,
                ),
            ),
            epsilon,
        )?;
        let feed_forward = match layer.feed_forward() {
            LagunaFeedForwardDescriptor::Dense(_) => reference_dense_feed_forward(
                runtime,
                tensors,
                layer_index,
                &normalized_after_attention,
            )?,
            LagunaFeedForwardDescriptor::Moe(descriptor) => reference_moe(
                runtime,
                tensors,
                layer_index,
                descriptor,
                &normalized_after_attention,
                contract.model().router_logit_softcap(),
            )?,
        };
        hidden = runtime.add(&after_attention, &feed_forward)?;
    }
    let normalized = runtime.rms_norm(
        &hidden,
        tensor(tensors, global(LagunaGlobalTensorRole::FinalNormalization)),
        epsilon,
    )?;
    // Serving projects only the terminal prompt position. Mirroring that public
    // boundary prevents this oracle from spending vocabulary work on positions
    // the sampler can never observe.
    let shape = normalized.shape();
    let terminal = runtime.slice(
        &normalized,
        &[0, shape[1] - 1, 0],
        &[shape[0], shape[1], shape[2]],
        &[1, 1, 1],
    )?;
    if contract.model().has_tied_embeddings() {
        let transposed = runtime.transpose_axes(embedding, &[1, 0])?;
        runtime.matmul(&terminal, &transposed)
    } else {
        linear(
            runtime,
            tensors,
            global(LagunaGlobalTensorRole::OutputHead),
            &terminal,
        )
    }
}

fn reference_dense_feed_forward(
    runtime: &MlxRuntime,
    tensors: &HashMap<LagunaTensorId, MlxArray>,
    layer_index: usize,
    hidden_states: &MlxArray,
) -> Result<MlxArray, MlxRuntimeError> {
    let project = |projection| {
        linear(
            runtime,
            tensors,
            layer_id(
                layer_index,
                LagunaLayerTensorRole::DenseFeedForward(projection),
            ),
            hidden_states,
        )
    };
    let gate = project(LagunaExpertProjection::Gate)?;
    let up = project(LagunaExpertProjection::Up)?;
    let activated = runtime.multiply(&runtime.silu(&gate)?, &up)?;
    linear(
        runtime,
        tensors,
        layer_id(
            layer_index,
            LagunaLayerTensorRole::DenseFeedForward(LagunaExpertProjection::Down),
        ),
        &activated,
    )
}

fn reference_attention(
    runtime: &MlxRuntime,
    tensors: &HashMap<LagunaTensorId, MlxArray>,
    layer_index: usize,
    descriptor: &astronomical_model_serving::LagunaAttentionDescriptor,
    hidden: &MlxArray,
    state: &mut ReferenceLayerState,
    epsilon: f32,
) -> Result<MlxArray, MlxRuntimeError> {
    let shape = hidden.shape();
    let batch = shape[0];
    let token_count = shape[1];
    let query_heads = descriptor.query_head_count() as i32;
    let key_value_heads = descriptor.key_value_head_count() as i32;
    let head_dimension = descriptor.head_dimension() as i32;
    let project = |projection, heads| -> Result<MlxArray, MlxRuntimeError> {
        let projected = linear(
            runtime,
            tensors,
            layer_id(layer_index, LagunaLayerTensorRole::Attention(projection)),
            hidden,
        )?;
        runtime.reshape(&projected, &[batch, token_count, heads, head_dimension])
    };
    let queries = runtime.rms_norm(
        &project(LagunaAttentionProjection::Query, query_heads)?,
        tensor(
            tensors,
            layer_id(
                layer_index,
                LagunaLayerTensorRole::AttentionQueryNormalization,
            ),
        ),
        epsilon,
    )?;
    let keys = runtime.rms_norm(
        &project(LagunaAttentionProjection::Key, key_value_heads)?,
        tensor(
            tensors,
            layer_id(
                layer_index,
                LagunaLayerTensorRole::AttentionKeyNormalization,
            ),
        ),
        epsilon,
    )?;
    let values = project(LagunaAttentionProjection::Value, key_value_heads)?;
    let queries = apply_rope(
        runtime,
        &runtime.transpose_axes(&queries, &[0, 2, 1, 3])?,
        descriptor.rope(),
        state.absolute_position,
    )?;
    let keys = apply_rope(
        runtime,
        &runtime.transpose_axes(&keys, &[0, 2, 1, 3])?,
        descriptor.rope(),
        state.absolute_position,
    )?;
    let values = runtime.transpose_axes(&values, &[0, 2, 1, 3])?;
    let previous_absolute_position = state.absolute_position;
    let active_keys = append(runtime, state.keys.as_ref(), &keys)?;
    let active_values = append(runtime, state.values.as_ref(), &values)?;
    let active_key_count = active_keys.shape()[2];
    let window_size = state
        .window_size
        .unwrap_or(active_key_count.saturating_add(1));
    let first_key_position = previous_absolute_position + token_count - active_key_count;
    // Use an explicit mask even for full attention. Production deliberately uses
    // MLX's native causal mode there, so this path can detect offset or cache
    // mistakes that would be shared by two calls to the same fast operation.
    let mask = build_causal_sliding_window_mask(
        runtime,
        previous_absolute_position,
        token_count,
        first_key_position,
        active_key_count,
        window_size,
        &mut PerformanceAttribution::disabled(),
    )?;
    let attended = runtime.masked_scaled_dot_product_attention(
        &queries,
        &active_keys,
        &active_values,
        (head_dimension as f32).sqrt().recip(),
        &mask,
    )?;
    state.absolute_position += token_count;
    let (committed_keys, committed_values) =
        commit_cache(runtime, &active_keys, &active_values, state.window_size)?;
    state.keys = Some(committed_keys);
    state.values = Some(committed_values);
    let token_major = runtime.transpose_axes(&attended, &[0, 2, 1, 3])?;
    let flattened = runtime.reshape(
        &token_major,
        &[batch, token_count, query_heads * head_dimension],
    )?;
    let gated = match descriptor.gating_kind() {
        LagunaGatingKind::None => flattened,
        gating_kind => {
            let gate_logits = linear(
                runtime,
                tensors,
                layer_id(
                    layer_index,
                    LagunaLayerTensorRole::Attention(LagunaAttentionProjection::Gate),
                ),
                hidden,
            )?;
            let gate_shape = match gating_kind {
                LagunaGatingKind::PerHead => vec![batch, token_count, query_heads, 1],
                LagunaGatingKind::PerElement => {
                    vec![batch, token_count, query_heads, head_dimension]
                }
                LagunaGatingKind::None => unreachable!(),
            };
            let shaped_logits = runtime.reshape(&gate_logits, &gate_shape)?;
            let shaped_output = runtime.reshape(
                &flattened,
                &[batch, token_count, query_heads, head_dimension],
            )?;
            let float_gate = runtime.astype(&shaped_logits, MlxDtype::Float32)?;
            let softplus_gate =
                runtime.astype(&runtime.softplus(&float_gate)?, shaped_output.dtype())?;
            runtime.reshape(
                &runtime.multiply(&shaped_output, &softplus_gate)?,
                &flattened.shape(),
            )?
        }
    };
    linear(
        runtime,
        tensors,
        layer_id(
            layer_index,
            LagunaLayerTensorRole::Attention(LagunaAttentionProjection::Output),
        ),
        &gated,
    )
}

fn apply_rope(
    runtime: &MlxRuntime,
    input: &MlxArray,
    rope: &LagunaRopeDescriptor,
    offset: i32,
) -> Result<MlxArray, MlxRuntimeError> {
    match rope {
        LagunaRopeDescriptor::Default(descriptor) => runtime.rope(
            input,
            descriptor.rotary_dimension() as i32,
            descriptor.rope_theta() as f32,
            offset,
        ),
        LagunaRopeDescriptor::Yarn(descriptor) => {
            let frequencies = compute_yarn_rope_frequency_denominators(
                descriptor.rope_theta(),
                descriptor.rotary_dimension(),
                descriptor.original_maximum_position_count(),
                descriptor.factor(),
                descriptor.beta_fast(),
                descriptor.beta_slow(),
            )
            .expect("reference YaRN frequencies should construct");
            let frequency_array = runtime.array_from_f32(
                frequencies.frequency_denominators(),
                &[frequencies.frequency_denominators().len() as i32],
            )?;
            let prepared = scale_rotary_prefix(
                runtime,
                input,
                descriptor.rotary_dimension() as i32,
                descriptor.attention_factor() as f32,
            )?;
            runtime.rope_with_custom_frequencies(
                &prepared,
                descriptor.rotary_dimension() as i32,
                &frequency_array,
                1.0,
                offset,
            )
        }
    }
}

fn scale_rotary_prefix(
    runtime: &MlxRuntime,
    input: &MlxArray,
    rotary_dimension: i32,
    factor: f32,
) -> Result<MlxArray, MlxRuntimeError> {
    let shape = input.shape();
    if rotary_dimension >= shape[3] {
        return runtime.multiply_scalar(input, factor);
    }
    let scaled = runtime.multiply_scalar(input, factor)?;
    let prefix = runtime.slice(
        &scaled,
        &[0, 0, 0, 0],
        &[shape[0], shape[1], shape[2], rotary_dimension],
        &[1, 1, 1, 1],
    )?;
    let tail = runtime.slice(input, &[0, 0, 0, rotary_dimension], &shape, &[1, 1, 1, 1])?;
    runtime.concatenate_axis(&[&prefix, &tail], 3)
}

fn append(
    runtime: &MlxRuntime,
    previous: Option<&MlxArray>,
    current: &MlxArray,
) -> Result<MlxArray, MlxRuntimeError> {
    previous.map_or_else(
        || current.retain(),
        |previous| runtime.concatenate_axis(&[previous, current], 2),
    )
}

fn commit_cache(
    runtime: &MlxRuntime,
    keys: &MlxArray,
    values: &MlxArray,
    window_size: Option<i32>,
) -> Result<(MlxArray, MlxArray), MlxRuntimeError> {
    let Some(window_size) = window_size else {
        return Ok((keys.retain()?, values.retain()?));
    };
    let token_count = keys.shape()[2];
    if token_count <= window_size {
        return Ok((keys.retain()?, values.retain()?));
    }
    // Keeping the chronological suffix is intentionally unlike production's
    // ring layout; attention is invariant when key/value pairs share ordering.
    let slice = |array: &MlxArray| {
        runtime.slice(
            array,
            &[0, 0, token_count - window_size, 0],
            &[
                array.shape()[0],
                array.shape()[1],
                token_count,
                array.shape()[3],
            ],
            &[1, 1, 1, 1],
        )
    };
    Ok((slice(keys)?, slice(values)?))
}

fn linear(
    runtime: &MlxRuntime,
    tensors: &HashMap<LagunaTensorId, MlxArray>,
    tensor_id: LagunaTensorId,
    input: &MlxArray,
) -> Result<MlxArray, MlxRuntimeError> {
    let weight = tensor(tensors, tensor_id);
    runtime.matmul(input, &runtime.transpose_axes(weight, &[1, 0])?)
}

fn tensor(tensors: &HashMap<LagunaTensorId, MlxArray>, tensor_id: LagunaTensorId) -> &MlxArray {
    tensors
        .get(&tensor_id)
        .unwrap_or_else(|| panic!("reference tensor {tensor_id:?} should exist"))
}

fn global(role: LagunaGlobalTensorRole) -> LagunaTensorId {
    LagunaTensorId::Global {
        role,
        component: LagunaTensorComponent::Weight,
    }
}

fn layer_id(layer_index: usize, role: LagunaLayerTensorRole) -> LagunaTensorId {
    LagunaTensorId::Layer {
        layer_index,
        role,
        component: LagunaTensorComponent::Weight,
    }
}
