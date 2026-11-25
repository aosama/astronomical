//! The target forward graph both Qwen3.5 text models build: token checks,
//! the embedding lookup, the decoder layers, and the vocabulary head.
//!
//! Only the decoder-layer loop differs between the engines. The resident model
//! runs layers it holds and may submit finished layer groups so macOS can
//! schedule other graphics work between command buffers. The streaming model
//! clears what a previous forward left, reads streamed layers and expert pages,
//! and submits each finished layer itself. Each model supplies its loop
//! through [`Qwen3_5TargetForwardGraph::build_decoder_layers`]; everything
//! around it is written here once.
//!
//! The trait is reached through static generics, so each model compiles its
//! own copy of these steps, exactly as when each model wrote them out.

use std::ops::Deref;

use mlx_bridge::{MlxArray, MlxDtype};
use runtime_integration::{MlxRuntime, MlxRuntimeError};

use crate::qwen3_5_core::model::Qwen3_5ModelBase;
use crate::qwen3_5_core::model_math::forward_contract;
use crate::{AstroDecoderStateCacheState, PerformanceAttribution, PerformanceOperation};
use text_generation::execution_error::{TextExecutionError, invalid_request_decoder_state};
use text_generation::{AstroSsdPromptCacheBoundaryCheckpointCollector, RequestDecoderStateStack};

/// Target-model graph outputs retained for specialized consumers.
pub struct Qwen3_5TargetForwardOutput {
    final_logits: MlxArray,
    pre_final_normalization_hidden_states: MlxArray,
}

impl Qwen3_5TargetForwardOutput {
    /// Returns float32 logits for the final target position.
    #[must_use]
    pub fn final_logits(&self) -> &MlxArray {
        &self.final_logits
    }

    /// Returns every final decoder row before the trunk's final RMS normalization.
    #[must_use]
    pub fn pre_final_normalization_hidden_states(&self) -> &MlxArray {
        &self.pre_final_normalization_hidden_states
    }

    /// Returns one pre-final-normalization hidden row from this target forward.
    pub fn pre_final_normalization_hidden_state_at(
        &self,
        runtime: &MlxRuntime,
        token_position_index: i32,
    ) -> Result<MlxArray, MlxRuntimeError> {
        pre_final_normalization_hidden_state_at(
            runtime,
            &self.pre_final_normalization_hidden_states,
            token_position_index,
        )
    }
}

/// One forward's position inputs, as the decoder layers read them.
#[derive(Clone, Copy)]
pub struct Qwen3_5ForwardPositions<'forward> {
    pub token_count: i32,
    pub starting_position_tokens: u32,
    pub token_position_offsets: Option<&'forward MlxArray>,
}

impl Qwen3_5ForwardPositions<'_> {
    /// The rotary position offset of the forward's first token.
    pub fn rope_offset_tokens(&self) -> Result<i32, TextExecutionError> {
        i32::try_from(self.starting_position_tokens).map_err(|_| TextExecutionError::InvalidInput {
            description: "starting position exceeds the MLX int32 range",
        })
    }
}

/// A Qwen3.5 text model's forward graph. A model supplies its decoder-layer
/// loop; the default methods build every kind of forward around it.
pub trait Qwen3_5TargetForwardGraph: Deref<Target = Qwen3_5ModelBase> {
    /// How a forward runs beyond its tokens: the streaming model's paged
    /// execution mode. The resident model runs every forward one way, so its
    /// mode is `()`.
    type ForwardMode: Copy;

    /// Runs every decoder layer over `hidden_states` and returns the last
    /// layer's output, before the final normalization.
    fn build_decoder_layers(
        &self,
        hidden_states: MlxArray,
        forward_positions: Qwen3_5ForwardPositions<'_>,
        request_decoder_state: &mut RequestDecoderStateStack,
        boundary_checkpoint_collector: Option<&mut AstroSsdPromptCacheBoundaryCheckpointCollector>,
        forward_mode: Self::ForwardMode,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, TextExecutionError>;

    /// The final-position logits of a forward over token indices.
    fn build_forward_graph(
        &self,
        token_indices: &MlxArray,
        token_count: i32,
        starting_position_tokens: u32,
        request_decoder_state: &mut RequestDecoderStateStack,
        forward_mode: Self::ForwardMode,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, TextExecutionError> {
        Ok(self
            .build_target_forward_graph_from_token_indices(
                token_indices,
                token_count,
                starting_position_tokens,
                request_decoder_state,
                None,
                forward_mode,
                performance_attribution,
            )?
            .final_logits)
    }

    /// Builds decoder-state prefill without the vocabulary head.
    ///
    /// Intermediate chunks only need `astro_decoder_state_cache` tensors; no
    /// `lm_head` runs until sampling. Building then dropping logits was
    /// host-side waste on every 2,048-token chunk.
    #[allow(clippy::too_many_arguments)]
    fn build_prefill_decoder_state_graph(
        &self,
        token_indices: &MlxArray,
        token_count: i32,
        starting_position_tokens: u32,
        request_decoder_state: &mut RequestDecoderStateStack,
        boundary_checkpoint_collector: Option<&mut AstroSsdPromptCacheBoundaryCheckpointCollector>,
        forward_mode: Self::ForwardMode,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(), TextExecutionError> {
        drop(
            self.build_target_forward_graph_from_token_indices_with_position_offsets(
                token_indices,
                Qwen3_5ForwardPositions {
                    token_count,
                    starting_position_tokens,
                    token_position_offsets: None,
                },
                request_decoder_state,
                boundary_checkpoint_collector,
                forward_mode,
                performance_attribution,
                false,
            )?,
        );
        Ok(())
    }

    fn build_target_forward_graph(
        &self,
        token_ids: &[u32],
        starting_position_tokens: u32,
        request_decoder_state: &mut RequestDecoderStateStack,
        boundary_checkpoint_collector: Option<&mut AstroSsdPromptCacheBoundaryCheckpointCollector>,
        forward_mode: Self::ForwardMode,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Qwen3_5TargetForwardOutput, TextExecutionError> {
        self.build_target_forward_graph_with_position_offsets(
            token_ids,
            starting_position_tokens,
            None,
            request_decoder_state,
            boundary_checkpoint_collector,
            forward_mode,
            performance_attribution,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn build_target_forward_graph_with_position_offsets(
        &self,
        token_ids: &[u32],
        starting_position_tokens: u32,
        token_position_offsets: Option<&MlxArray>,
        request_decoder_state: &mut RequestDecoderStateStack,
        boundary_checkpoint_collector: Option<&mut AstroSsdPromptCacheBoundaryCheckpointCollector>,
        forward_mode: Self::ForwardMode,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Qwen3_5TargetForwardOutput, TextExecutionError> {
        let (token_indices, token_count) = self.validated_token_indices(
            token_ids,
            starting_position_tokens,
            token_position_offsets,
            request_decoder_state,
        )?;
        self.build_target_forward_graph_from_token_indices_with_position_offsets(
            &token_indices,
            Qwen3_5ForwardPositions {
                token_count,
                starting_position_tokens,
                token_position_offsets,
            },
            request_decoder_state,
            boundary_checkpoint_collector,
            forward_mode,
            performance_attribution,
            true,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn build_target_forward_graph_from_token_indices(
        &self,
        token_indices: &MlxArray,
        token_count: i32,
        starting_position_tokens: u32,
        request_decoder_state: &mut RequestDecoderStateStack,
        boundary_checkpoint_collector: Option<&mut AstroSsdPromptCacheBoundaryCheckpointCollector>,
        forward_mode: Self::ForwardMode,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Qwen3_5TargetForwardOutput, TextExecutionError> {
        self.build_target_forward_graph_from_token_indices_with_position_offsets(
            token_indices,
            Qwen3_5ForwardPositions {
                token_count,
                starting_position_tokens,
                token_position_offsets: None,
            },
            request_decoder_state,
            boundary_checkpoint_collector,
            forward_mode,
            performance_attribution,
            true,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn build_target_forward_graph_from_token_indices_with_position_offsets(
        &self,
        token_indices: &MlxArray,
        forward_positions: Qwen3_5ForwardPositions<'_>,
        request_decoder_state: &mut RequestDecoderStateStack,
        boundary_checkpoint_collector: Option<&mut AstroSsdPromptCacheBoundaryCheckpointCollector>,
        forward_mode: Self::ForwardMode,
        performance_attribution: &mut PerformanceAttribution,
        should_materialize_vocabulary_logits: bool,
    ) -> Result<Qwen3_5TargetForwardOutput, TextExecutionError> {
        let hidden_states = self.embedding_lookup(token_indices)?;
        self.build_target_forward_graph_from_embeddings_with_position_offsets(
            hidden_states,
            forward_positions,
            request_decoder_state,
            boundary_checkpoint_collector,
            forward_mode,
            performance_attribution,
            should_materialize_vocabulary_logits,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn build_forward_graph_from_embeddings(
        &self,
        hidden_states: MlxArray,
        token_count: i32,
        starting_position_tokens: u32,
        request_decoder_state: &mut RequestDecoderStateStack,
        boundary_checkpoint_collector: Option<&mut AstroSsdPromptCacheBoundaryCheckpointCollector>,
        forward_mode: Self::ForwardMode,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, TextExecutionError> {
        Ok(self
            .build_target_forward_graph_from_embeddings(
                hidden_states,
                token_count,
                starting_position_tokens,
                request_decoder_state,
                boundary_checkpoint_collector,
                forward_mode,
                performance_attribution,
            )?
            .final_logits)
    }

    #[allow(clippy::too_many_arguments)]
    fn build_target_forward_graph_from_embeddings(
        &self,
        hidden_states: MlxArray,
        token_count: i32,
        starting_position_tokens: u32,
        request_decoder_state: &mut RequestDecoderStateStack,
        boundary_checkpoint_collector: Option<&mut AstroSsdPromptCacheBoundaryCheckpointCollector>,
        forward_mode: Self::ForwardMode,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Qwen3_5TargetForwardOutput, TextExecutionError> {
        self.build_target_forward_graph_from_embeddings_with_position_offsets(
            hidden_states,
            Qwen3_5ForwardPositions {
                token_count,
                starting_position_tokens,
                token_position_offsets: None,
            },
            request_decoder_state,
            boundary_checkpoint_collector,
            forward_mode,
            performance_attribution,
            true,
        )
    }

    /// Runs the decoder layers, then the vocabulary head unless only the
    /// decoder state is wanted.
    #[allow(clippy::too_many_arguments)]
    fn build_target_forward_graph_from_embeddings_with_position_offsets(
        &self,
        hidden_states: MlxArray,
        forward_positions: Qwen3_5ForwardPositions<'_>,
        request_decoder_state: &mut RequestDecoderStateStack,
        boundary_checkpoint_collector: Option<&mut AstroSsdPromptCacheBoundaryCheckpointCollector>,
        forward_mode: Self::ForwardMode,
        performance_attribution: &mut PerformanceAttribution,
        should_materialize_vocabulary_logits: bool,
    ) -> Result<Qwen3_5TargetForwardOutput, TextExecutionError> {
        let hidden_states = self.build_decoder_layers(
            hidden_states,
            forward_positions,
            request_decoder_state,
            boundary_checkpoint_collector,
            forward_mode,
            performance_attribution,
        )?;
        if !should_materialize_vocabulary_logits {
            return Ok(Qwen3_5TargetForwardOutput {
                final_logits: hidden_states.retain().map_err(TextExecutionError::from)?,
                pre_final_normalization_hidden_states: hidden_states,
            });
        }
        let final_logits = performance_attribution.measure_operation(
            PerformanceOperation::FinalLogitsGraphConstruction,
            |_performance_attribution| {
                self.final_vocabulary_logits(&hidden_states, forward_positions.token_count)
            },
        )?;
        Ok(Qwen3_5TargetForwardOutput {
            final_logits,
            pre_final_normalization_hidden_states: hidden_states,
        })
    }
}

impl Qwen3_5ModelBase {
    /// Checks one forward's token ids against the model and the decoder state,
    /// and returns them as the `[1, token_count]` index array the graph reads.
    pub fn validated_token_indices(
        &self,
        token_ids: &[u32],
        starting_position_tokens: u32,
        token_position_offsets: Option<&MlxArray>,
        request_decoder_state: &RequestDecoderStateStack,
    ) -> Result<(MlxArray, i32), TextExecutionError> {
        let token_count = forward_contract::validate_forward_input(
            token_ids,
            starting_position_tokens,
            token_position_offsets,
            request_decoder_state.layer_count(),
            self.config.layer_count() as usize,
            self.config.vocabulary_size(),
            self.config.maximum_position_count(),
        )?;
        let mut signed_token_ids = Vec::with_capacity(token_ids.len());
        for token_id in token_ids {
            signed_token_ids.push(i32::try_from(*token_id).map_err(|_| {
                TextExecutionError::InvalidInput {
                    description: "token ID exceeds the MLX int32 range",
                }
            })?);
        }
        let token_indices = self
            .runtime
            .array_from_i32(&signed_token_ids, &[1, token_count])?;
        Ok((token_indices, token_count))
    }

    /// Builds the final vocabulary logits for the last token position:
    /// normalize, slice the final token, project through the language-model
    /// head, and widen to Float32 for host sampling.
    pub fn final_vocabulary_logits(
        &self,
        hidden_states: &MlxArray,
        token_count: i32,
    ) -> Result<MlxArray, TextExecutionError> {
        let normalized_states = self.runtime.rms_norm(
            hidden_states,
            &self.weights.final_normalization_weight,
            f32::from_bits(self.config.rms_norm_epsilon_bits()),
        )?;
        let hidden_size = self.config.hidden_size() as i32;
        let final_hidden_state = self.runtime.slice(
            &normalized_states,
            &[0, token_count - 1, 0],
            &[1, token_count, hidden_size],
            &[1, 1, 1],
        )?;
        let final_logits = self.quantized_linear(
            &final_hidden_state,
            &self.weights.language_model_head_weights,
        )?;
        self.runtime
            .astype(&final_logits, MlxDtype::Float32)
            .map_err(TextExecutionError::from)
    }

    /// How many decoder layers a forward runs before submitting the finished
    /// group, or zero for none. `is_ssd_paging` selects the streaming
    /// engine's one-token pacing.
    pub fn graph_submission_layer_interval(&self, token_count: i32, is_ssd_paging: bool) -> usize {
        usize::try_from(ipc_protocol::graph_submission_layer_interval(
            token_count,
            is_ssd_paging,
            self.chunking.prefill_graph_submission_layer_interval,
            self.chunking
                .experimental_ssd_paging_generation_graph_submission_layer_interval,
        ))
        .unwrap_or(0)
    }
}

/// The decoder state one layer reads and writes, refused when it is missing
/// or its tensors are only partly allocated.
pub fn validated_layer_decoder_state(
    request_decoder_state: &mut RequestDecoderStateStack,
    layer_index: usize,
) -> Result<&mut AstroDecoderStateCacheState, TextExecutionError> {
    let layer_model_state = request_decoder_state.layer_mut(layer_index).ok_or(
        TextExecutionError::InvalidRequestDecoderState {
            layer_index,
            description: "decoder layer entry is missing",
        },
    )?;
    if !layer_model_state.tensors_are_allocated_consistently() {
        return Err(invalid_request_decoder_state(
            layer_index,
            "decoder layer tensors must be allocated together or absent together",
        ));
    }
    Ok(layer_model_state)
}

pub fn pre_final_normalization_hidden_state_at(
    runtime: &MlxRuntime,
    pre_final_normalization_hidden_states: &MlxArray,
    token_position_index: i32,
) -> Result<MlxArray, MlxRuntimeError> {
    let hidden_state_shape = pre_final_normalization_hidden_states.shape();
    if hidden_state_shape.len() != 3
        || hidden_state_shape[0] != 1
        || token_position_index < 0
        || token_position_index >= hidden_state_shape[1]
    {
        return Err(MlxRuntimeError::RuntimeOperation {
            operation: "slice Qwen3.5 pre-final hidden state",
            description: "hidden-state row index is outside the target forward output".to_owned(),
        });
    }
    Ok(runtime.slice(
        pre_final_normalization_hidden_states,
        &[0, token_position_index, 0],
        &[1, token_position_index + 1, hidden_state_shape[2]],
        &[1, 1, 1],
    )?)
}
