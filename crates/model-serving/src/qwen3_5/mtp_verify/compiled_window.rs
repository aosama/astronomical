//! The compiled verification-window lane: cache, geometry, apply, install.
//!
//! One loaded model owns one compiled graph per window row count (two through
//! four), traced on first use and replayed afterwards. A window apply
//! assembles the frozen compile-input vector from the model's resident weights
//! and the request's state leaves, replays the graph, reads the ordered
//! outputs, and installs every state successor through the eager state
//! owners' own public install paths — full-attention rotated keys and values
//! go through the append-only owner's update path so the eager
//! capacity-growth policy keeps owning that state.
//!
//! Every refusal is a decline, not an error in the eager pipeline: the caller
//! falls back to the eager verification window.

use std::cell::RefCell;
use std::sync::Arc;

use astronomical_runtime_integration::{
    MlxCompiledVerifyWindowGraph, VerifyWindowFullAttentionQuantization,
    VerifyWindowGatedDeltaQuantization, VerifyWindowGdnKernelSet, VerifyWindowGeometry,
    VerifyWindowLayerKind, VerifyWindowLayerQuantization, VerifyWindowQuantizationPair,
    VerifyWindowTrunkQuantization,
};

use crate::decoder_cache::{DecoderCacheLayerLayout, DecoderCacheState};
use crate::qwen3_5::decoder::RequestDecoderStateStack;
use crate::qwen3_5::model::Qwen3_5Model;
use crate::qwen3_5::model::decoder_layer_weights::{
    Qwen3_5AffineWeights, Qwen3_5AttentionWeights, Qwen3_5DecoderFeedForwardWeights,
    Qwen3_5DecoderLayerWeights,
};
use crate::qwen3_5::mtp_verify::window_abi_assembly::{
    VerifyWindowAttemptInputs, assemble_window_inputs, read_window_outputs,
};
use crate::qwen3_5::mtp_verify::window_state_leaves::{
    WindowStateLeaves, WindowStateUpdate, extract_window_input_leaves, install_window_output_leaves,
};
use astronomical_mlx_c_rust::MlxArray;

/// The supported window row counts (draft depth plus one).
pub(crate) const SUPPORTED_ROW_COUNTS: [i32; 3] = [2, 3, 4];

/// One model's compiled verification windows, one per row count.
#[derive(Debug, Default)]
pub(crate) struct MtpVerifyWindowLane {
    compiled_windows: RefCell<[Option<Arc<MlxCompiledVerifyWindowGraph>>; 3]>,
}

impl MtpVerifyWindowLane {
    fn slot_index(row_count: i32) -> Option<usize> {
        SUPPORTED_ROW_COUNTS
            .iter()
            .position(|supported_row_count| *supported_row_count == row_count)
    }

    /// Returns the retained compiled window for one row count, compiling it
    /// on first use.
    fn compiled_window_for(
        &self,
        model: &Qwen3_5Model,
        row_count: i32,
    ) -> Result<Arc<MlxCompiledVerifyWindowGraph>, String> {
        let slot_index = Self::slot_index(row_count)
            .ok_or_else(|| format!("row count {row_count} has no compiled window slot"))?;
        if let Some(compiled_window) = self.compiled_windows.borrow()[slot_index].as_ref() {
            return Ok(Arc::clone(compiled_window));
        }
        let geometry = verify_window_geometry(model, row_count)?;
        let gdn_kernels = verify_window_gdn_kernels(model);
        let compiled_window = Arc::new(
            MlxCompiledVerifyWindowGraph::new(geometry, gdn_kernels)
                .map_err(|error| error.to_string())?,
        );
        let mut compiled_windows = self.compiled_windows.borrow_mut();
        let slot = compiled_windows[slot_index].get_or_insert_with(|| Arc::clone(&compiled_window));
        Ok(Arc::clone(slot))
    }
}

/// Builds the compiled window's geometry from one model's validated facts,
/// refusing non-uniform quantization the window does not support yet.
pub(crate) fn verify_window_geometry(
    model: &Qwen3_5Model,
    row_count: i32,
) -> Result<VerifyWindowGeometry, String> {
    let config = model.config();
    let decoder_cache_layout = model.decoder_cache_layout();
    if config.layer_count() as usize != decoder_cache_layout.layer_count() {
        return Err(format!(
            "decoder-cache layout has {} layers for {} configured layers",
            decoder_cache_layout.layer_count(),
            config.layer_count()
        ));
    }
    let mut layer_kinds = Vec::with_capacity(decoder_cache_layout.layer_count());
    for layer_index in 0..decoder_cache_layout.layer_count() {
        let layer_layout = decoder_cache_layout
            .layer(layer_index)
            .ok_or_else(|| format!("decoder layer {layer_index} has no cache layout"))?;
        let layer_kind = match layer_layout {
            DecoderCacheLayerLayout::Composite { .. } => VerifyWindowLayerKind::GatedDelta,
            DecoderCacheLayerLayout::AppendOnlyAttention { .. } => {
                VerifyWindowLayerKind::FullAttention
            }
            other => {
                return Err(format!(
                    "decoder layer {layer_index} has cache layout {other:?} without a verification-window state"
                ));
            }
        };
        layer_kinds.push(layer_kind);
    }
    let mut layer_quantization = Vec::with_capacity(model.weights.decoder_layer_weights.len());
    for (layer_index, decoder_layer_weights) in
        model.weights.decoder_layer_weights.iter().enumerate()
    {
        layer_quantization.push(layer_quantization_pair(layer_index, decoder_layer_weights)?);
    }
    let embedding_pair = affine_quantization_pair(&model.weights.embedding_weights)?;
    let language_model_head_pair =
        affine_quantization_pair(&model.weights.language_model_head_weights)?;
    Ok(VerifyWindowGeometry::new(
        row_count,
        layer_kinds,
        config.query_head_count() as i32,
        config.key_value_head_count() as i32,
        config.head_dimension() as i32,
        config.rotary_dimension() as i32,
        config.linear_key_head_count() as i32,
        config.linear_value_head_count() as i32,
        config.linear_key_head_dimension() as i32,
        config.linear_key_dimension() as i32,
        config.linear_convolution_dimension() as i32,
        config.linear_convolution_kernel_dimension() as i32,
        layer_quantization,
        VerifyWindowTrunkQuantization {
            embedding: embedding_pair,
            language_model_head: language_model_head_pair,
        },
        f32::from_bits(config.rms_norm_epsilon_bits()),
        f32::from_bits(config.rope_theta_bits()),
    ))
}

/// Every quantized pair one decoder layer consumes, shaped by its attention
/// family.
fn layer_quantization_pair(
    layer_index: usize,
    decoder_layer_weights: &Qwen3_5DecoderLayerWeights,
) -> Result<VerifyWindowLayerQuantization, String> {
    let feed_forward_quantization =
        feed_forward_quantization_pair(layer_index, decoder_layer_weights)?;
    match &decoder_layer_weights.attention_weights {
        Qwen3_5AttentionWeights::Linear(linear_attention_weights) => {
            Ok(VerifyWindowLayerQuantization {
                gated_delta: Some(VerifyWindowGatedDeltaQuantization {
                    input_queries_keys_values: affine_quantization_pair(
                        &linear_attention_weights.input_queries_keys_values_projection,
                    )?,
                    output_gate: affine_quantization_pair(
                        &linear_attention_weights.output_gate_projection,
                    )?,
                    update_rate: affine_quantization_pair(
                        &linear_attention_weights.update_rate_projection,
                    )?,
                    decay_interval: affine_quantization_pair(
                        &linear_attention_weights.decay_interval_projection,
                    )?,
                    output_projection: affine_quantization_pair(
                        &linear_attention_weights.output_projection,
                    )?,
                    feed_forward_gate: feed_forward_quantization.0,
                    feed_forward_up: feed_forward_quantization.1,
                    feed_forward_down: feed_forward_quantization.2,
                }),
                full_attention: None,
            })
        }
        Qwen3_5AttentionWeights::Full(full_attention_weights) => {
            Ok(VerifyWindowLayerQuantization {
                gated_delta: None,
                full_attention: Some(VerifyWindowFullAttentionQuantization {
                    query: affine_quantization_pair(&full_attention_weights.query_projection)?,
                    key: affine_quantization_pair(&full_attention_weights.key_projection)?,
                    value: affine_quantization_pair(&full_attention_weights.value_projection)?,
                    output: affine_quantization_pair(&full_attention_weights.output_projection)?,
                    feed_forward_gate: feed_forward_quantization.0,
                    feed_forward_up: feed_forward_quantization.1,
                    feed_forward_down: feed_forward_quantization.2,
                }),
            })
        }
    }
}

fn feed_forward_quantization_pair(
    layer_index: usize,
    decoder_layer_weights: &Qwen3_5DecoderLayerWeights,
) -> Result<
    (
        VerifyWindowQuantizationPair,
        VerifyWindowQuantizationPair,
        VerifyWindowQuantizationPair,
    ),
    String,
> {
    let Qwen3_5DecoderFeedForwardWeights::Dense(dense_mlp_weights) =
        &decoder_layer_weights.mlp_weights
    else {
        return Err(format!(
            "decoder layer {layer_index} has no dense feed-forward weights for the compiled window"
        ));
    };
    Ok((
        affine_quantization_pair(&dense_mlp_weights.gate_projection)?,
        affine_quantization_pair(&dense_mlp_weights.up_projection)?,
        affine_quantization_pair(&dense_mlp_weights.down_projection)?,
    ))
}

fn affine_quantization_pair(
    affine_weights: &Qwen3_5AffineWeights,
) -> Result<VerifyWindowQuantizationPair, String> {
    let Qwen3_5AffineWeights::Quantized {
        quantization_group_size,
        quantization_bits,
        ..
    } = affine_weights
    else {
        return Err("a compiled-window affine module is not quantized".to_owned());
    };
    Ok(VerifyWindowQuantizationPair {
        group_size: *quantization_group_size,
        bits: *quantization_bits,
    })
}

/// One compiled window apply's outputs, with every state successor already
/// installed into the request's state stack.
pub(crate) struct CompiledVerificationWindow {
    pub(crate) all_position_logits: MlxArray,
    pub(crate) pre_final_normalization_hidden_states: MlxArray,
    pub(crate) boundary_collector:
        Option<crate::qwen3_5::decoder::Qwen3_5PersistentPromptCacheBoundaryCheckpointCollector>,
}

impl Qwen3_5Model {
    /// Runs one compiled verification window and installs its state.
    ///
    /// Declines with `Err` (the caller falls back to the eager window) when
    /// the artifact or state does not fit the compiled lane's contract.
    pub(crate) fn run_compiled_verification_window(
        &self,
        token_ids: &[u32],
        starting_position_tokens: u32,
        request_decoder_state: &mut RequestDecoderStateStack,
    ) -> Result<CompiledVerificationWindow, String> {
        let row_count = i32::try_from(token_ids.len())
            .map_err(|_| "window row count exceeds the int32 range".to_owned())?;
        if !SUPPORTED_ROW_COUNTS.contains(&row_count) {
            return Err(format!("window row count {row_count} is not supported"));
        }
        if self.sparse_experts_are_paged() {
            return Err("paged sparse experts have no compiled window".to_owned());
        }
        let compiled_window = self.mtp_verify_lane.compiled_window_for(self, row_count)?;
        let mut signed_token_ids = Vec::with_capacity(token_ids.len());
        for token_id in token_ids {
            signed_token_ids.push(
                i32::try_from(*token_id)
                    .map_err(|_| "token ID exceeds the int32 range".to_owned())?,
            );
        }
        let token_indices = self
            .runtime
            .array_from_i32(&signed_token_ids, &[1, row_count])
            .map_err(|error| error.to_string())?;
        let starting_position = i32::try_from(starting_position_tokens)
            .map_err(|_| "starting position exceeds the int32 range".to_owned())?;
        let window_positions: Vec<i32> = (0..row_count)
            .map(|row_index| starting_position + row_index)
            .collect();
        let position_offsets = self
            .runtime
            .array_from_i32(&window_positions, &[row_count])
            .map_err(|error| error.to_string())?;
        let state_leaves = extract_window_input_leaves(request_decoder_state)?;
        let key_value_base_offset = first_full_attention_offset(&state_leaves)?;
        let key_value_base_offset_array = self
            .runtime
            .array_from_i32(&[key_value_base_offset], &[])
            .map_err(|error| error.to_string())?;
        let attempt_inputs = VerifyWindowAttemptInputs {
            token_indices: &token_indices,
            position_offsets: &position_offsets,
            key_value_base_offset: &key_value_base_offset_array,
            query_normalization_scale: &self.query_normalization_scale_weight,
            key_normalization_scale: &self.key_normalization_scale_weight,
            state_leaves: &state_leaves,
        };
        let graph_inputs = assemble_window_inputs(
            compiled_window.geometry(),
            &self.weights.decoder_layer_weights,
            &self.weights.embedding_weights,
            &self.weights.final_normalization_weight,
            &self.weights.language_model_head_weights,
            &attempt_inputs,
        )?;
        let graph_input_references = graph_inputs.iter().collect::<Vec<_>>();
        let window_outputs = self
            .runtime
            .apply_compiled_verify_window_graph(
                &compiled_window,
                &graph_input_references,
                verify_window_gdn_kernels(self),
            )
            .map_err(|error| error.to_string())?;
        let attempt_outputs = read_window_outputs(compiled_window.geometry(), window_outputs)?;
        install_window_output_leaves(request_decoder_state, &attempt_outputs.layer_updates)?;
        install_full_attention_windows(
            self,
            request_decoder_state,
            &attempt_outputs.layer_updates,
            key_value_base_offset,
        )?;
        let boundary_collector = record_compiled_boundary_snapshots(
            self,
            compiled_window.geometry(),
            attempt_outputs.layer_updates,
        )?;
        Ok(CompiledVerificationWindow {
            all_position_logits: attempt_outputs.all_position_logits,
            pre_final_normalization_hidden_states: attempt_outputs
                .pre_final_normalization_hidden_states,
            boundary_collector,
        })
    }

    /// Test-facing compiled window forward returning materialized float32
    /// all-position logits.
    #[doc(hidden)]
    pub fn compiled_verification_window_logits_for_tests(
        &self,
        token_ids: &[u32],
        starting_position_tokens: u32,
        request_decoder_state: &mut RequestDecoderStateStack,
    ) -> Result<MlxArray, String> {
        let compiled_window = self.run_compiled_verification_window(
            token_ids,
            starting_position_tokens,
            request_decoder_state,
        )?;
        self.runtime
            .evaluate_arrays(&[&compiled_window.all_position_logits])
            .map_err(|error| error.to_string())?;
        Ok(compiled_window.all_position_logits)
    }
}

fn verify_window_gdn_kernels(model: &Qwen3_5Model) -> VerifyWindowGdnKernelSet<'_> {
    VerifyWindowGdnKernelSet {
        prework_kernel: model.gdn_decode_prework_kernel.as_ref(),
        checkpoint_kernel: model.gated_delta_checkpoint_kernel.as_ref(),
        query_normalization_scale: &model.inverse_linear_head_dimension_scale,
        key_normalization_scale: &model.inverse_square_root_linear_head_dimension_scale,
        prework_lane_count: crate::qwen3_5::model::GDN_PREWORK_LANE_COUNT,
        checkpoint_threadgroup_thread_count:
            crate::qwen3_5::model::GDN_CHECKPOINT_THREADGROUP_THREAD_COUNT as i32,
        checkpoint_value_row_block_size: crate::qwen3_5::model::GDN_CHECKPOINT_VALUE_ROW_BLOCK_SIZE
            as i32,
    }
}

/// Feeds the compiled window's per-row boundary snapshots into the same
/// collector contract the eager window satisfies, so verifier-prefix rollback
/// and persistent-cache state behave identically.
fn record_compiled_boundary_snapshots(
    model: &Qwen3_5Model,
    geometry: &VerifyWindowGeometry,
    layer_updates: Vec<WindowStateUpdate>,
) -> Result<
    Option<crate::qwen3_5::decoder::Qwen3_5PersistentPromptCacheBoundaryCheckpointCollector>,
    String,
> {
    let boundary_row_count = geometry.row_count().saturating_sub(1).max(0) as usize;
    if boundary_row_count == 0 {
        return Ok(None);
    }
    let recurrent_boundary_tensor_count = model.decoder_cache_layout().boundary_tensor_count();
    if recurrent_boundary_tensor_count == 0 {
        return Ok(None);
    }
    let completed_boundary_rows = (1..=boundary_row_count as i32)
        .map(|row| row as usize)
        .collect::<Vec<_>>();
    let mut boundary_collector =
        crate::qwen3_5::decoder::Qwen3_5PersistentPromptCacheBoundaryCheckpointCollector::new(
            completed_boundary_rows,
            recurrent_boundary_tensor_count,
            1,
        )
        .map_err(|error| error.to_string())?;
    for (layer_index, layer_update) in layer_updates.into_iter().enumerate() {
        let WindowStateUpdate::GatedDelta {
            boundary_convolution_states,
            boundary_recurrent_states,
            ..
        } = layer_update
        else {
            continue;
        };
        boundary_collector
            .record_linear_attention_layer(
                layer_index,
                boundary_convolution_states,
                boundary_recurrent_states,
            )
            .map_err(|error| error.to_string())?;
    }
    Ok(Some(boundary_collector))
}

/// The logical key/value offset carried by the extracted full-attention
/// leaves — the offset every layer's slab install appends from.
fn first_full_attention_offset(state_leaves: &[WindowStateLeaves]) -> Result<i32, String> {
    for state_leaf in state_leaves {
        if let WindowStateLeaves::FullAttention { offset_tokens, .. } = state_leaf {
            return Ok(*offset_tokens);
        }
    }
    Err("the extracted leaves have no full-attention layer".to_owned())
}

/// Installs each full-attention layer's rotated keys and values through the
/// eager owner's own append path, preserving its capacity-growth policy.
fn install_full_attention_windows(
    model: &Qwen3_5Model,
    request_decoder_state: &mut RequestDecoderStateStack,
    layer_updates: &[WindowStateUpdate],
    previous_offset_tokens: i32,
) -> Result<(), String> {
    let runtime = model.runtime();
    for (layer_index, layer_update) in layer_updates.into_iter().enumerate() {
        let WindowStateUpdate::FullAttention {
            keys: rotated_keys,
            values,
        } = layer_update
        else {
            continue;
        };
        let Some(DecoderCacheState::AppendOnlyAttention { attention }) =
            request_decoder_state.layer_mut(layer_index)
        else {
            continue;
        };
        attention
            .update_and_fetch(runtime, rotated_keys, values, previous_offset_tokens)
            .map_err(|error| format!("full-attention install failed: {error}"))?;
    }
    Ok(())
}
