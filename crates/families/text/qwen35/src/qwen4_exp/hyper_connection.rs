//! Flash-Next's multi-lane residual and the hyper-connections that read it.
//!
//! Instead of one residual vector per token, Flash-Next carries `lanes`
//! of them side by side (`[.., lanes * hidden]`). Before each attention and
//! each expert block, a hyper-connection normalizes every lane, mixes the
//! lanes down to one hidden state for the block, and scores how strongly
//! the block's output is written back to each lane. The model's last mixer
//! only mixes, and its output feeds the vocabulary head directly; it is why
//! the checkpoint has no final norm.

use std::collections::HashMap;

use mlx_bridge::MlxArray;
use text_generation::execution_error::TextExecutionError;

use crate::qwen3_5_core::configuration::{Qwen3_5Config, Qwen4ExpArchitecture};
use crate::qwen3_5_core::model::Qwen3_5ModelBase;
use crate::qwen3_5_core::model_math::dense_projection::Qwen3_5DenseProjection;
use crate::qwen3_5_core::model_math::weights::take_dense_projection;

use super::centered_norm::grouped_rms_norm;

/// One hyper-connection's projections. Its lane norm is bound where the
/// Qwen 3.5 norm it replaces is: the layer's input or post-attention norm,
/// or the model's final norm.
#[derive(Debug)]
pub struct HyperConnectionWeights {
    pub mix_down: Qwen3_5DenseProjection,
    pub mix_up: Qwen3_5DenseProjection,
    /// Scores each lane's share of the block output; absent on the final
    /// mixer, which writes nothing back.
    pub write_back: Option<Qwen3_5DenseProjection>,
}

/// What a hyper-connection hands a block, and keeps for writing back.
pub struct LaneRead {
    /// The block's input: the lanes mixed down to one hidden state.
    pub block_input: MlxArray,
    /// The lanes as they were before the block.
    pub lanes: MlxArray,
    /// `[.., lanes]`: how much of the block output each lane receives.
    pub write_back_weights: MlxArray,
}

impl HyperConnectionWeights {
    /// Takes the connection under `prefix` out of the bound tensors.
    pub fn take(
        bound_tensors: &mut HashMap<String, MlxArray>,
        config: &Qwen3_5Config,
        prefix: &str,
        writes_back_to_lanes: bool,
    ) -> Result<Self, TextExecutionError> {
        Ok(Self {
            mix_down: take_dense_projection(
                bound_tensors,
                config,
                &format!("{prefix}.input_mix_weight_down"),
            )?,
            mix_up: take_dense_projection(
                bound_tensors,
                config,
                &format!("{prefix}.input_mix_weight_up"),
            )?,
            write_back: writes_back_to_lanes
                .then(|| {
                    take_dense_projection(
                        bound_tensors,
                        config,
                        &format!("{prefix}.block_inject_weight"),
                    )
                })
                .transpose()?,
        })
    }

    pub fn append_array_references<'weights>(&'weights self, arrays: &mut Vec<&'weights MlxArray>) {
        self.mix_down.append_array_references(arrays);
        self.mix_up.append_array_references(arrays);
        if let Some(write_back) = &self.write_back {
            write_back.append_array_references(arrays);
        }
    }
}

impl Qwen3_5ModelBase {
    /// Normalizes the lanes and mixes them into a block's input.
    ///
    /// `mix = silu(down(norm(lanes)) / lanes)`, then each lane is weighted by
    /// `sigmoid(up(mix))` and the lanes are averaged, as the reference does.
    pub fn read_lanes(
        &self,
        lanes: &MlxArray,
        lane_norm_scale: &MlxArray,
        connection: &HyperConnectionWeights,
        architecture: &Qwen4ExpArchitecture,
    ) -> Result<(MlxArray, MlxArray), TextExecutionError> {
        let hidden_size = self.config.hidden_size() as i32;
        let lane_count = architecture.residual_lane_count as i32;
        let normalized = grouped_rms_norm(
            &self.runtime,
            lanes,
            lane_norm_scale,
            hidden_size,
            f32::from_bits(self.config.rms_norm_epsilon_bits()),
        )?;
        let projected_down = self.quantized_linear(&normalized, &connection.mix_down)?;
        let mix = self.runtime.silu(
            &self
                .runtime
                .multiply_scalar(&projected_down, 1.0 / lane_count as f32)?,
        )?;
        let lane_weights = self
            .runtime
            .sigmoid(&self.quantized_linear(&mix, &connection.mix_up)?)?;
        let lane_shape = lane_split_shape(&normalized.shape(), lane_count, hidden_size);
        let weighted_lanes = self.runtime.multiply(
            &self.runtime.reshape(&lane_weights, &lane_shape)?,
            &self.runtime.reshape(&normalized, &lane_shape)?,
        )?;
        let block_input = self.runtime.mean_axis(&weighted_lanes, -2, false)?;
        Ok((block_input, normalized))
    }

    /// Reads the lanes for a decoder-layer block, with write-back weights.
    pub fn read_lanes_for_block(
        &self,
        lanes: &MlxArray,
        lane_norm_scale: &MlxArray,
        connection: &HyperConnectionWeights,
        architecture: &Qwen4ExpArchitecture,
    ) -> Result<LaneRead, TextExecutionError> {
        let write_back =
            connection
                .write_back
                .as_ref()
                .ok_or(TextExecutionError::InvalidInput {
                    description: "a decoder-layer hyper-connection must write back to its lanes",
                })?;
        let (block_input, normalized) =
            self.read_lanes(lanes, lane_norm_scale, connection, architecture)?;
        // `2 * sigmoid(score / lanes)` lets a lane take up to twice the output.
        let scores = self.quantized_linear(&normalized, write_back)?;
        let lane_count = architecture.residual_lane_count as f32;
        let write_back_weights = self.runtime.multiply_scalar(
            &self
                .runtime
                .sigmoid(&self.runtime.multiply_scalar(&scores, 1.0 / lane_count)?)?,
            2.0,
        )?;
        Ok(LaneRead {
            block_input,
            lanes: lanes.retain()?,
            write_back_weights,
        })
    }

    /// Adds `block_output` to every lane, weighted per lane.
    pub fn write_back_to_lanes(
        &self,
        block_output: &MlxArray,
        lane_read: &LaneRead,
    ) -> Result<MlxArray, TextExecutionError> {
        let lanes_shape = lane_read.lanes.shape();
        let block_output = self.runtime.expand_dims(block_output, -2)?;
        let weights = self
            .runtime
            .expand_dims(&lane_read.write_back_weights, -1)?;
        let injected = self.runtime.multiply(&block_output, &weights)?;
        Ok(self.runtime.add(
            &lane_read.lanes,
            &self.runtime.reshape(&injected, &lanes_shape)?,
        )?)
    }

    /// The last token's vocabulary logits, float32: the final mixer folds
    /// the lanes into one hidden state, which feeds the head directly.
    pub fn lane_mixed_vocabulary_logits(
        &self,
        lanes: &MlxArray,
        lane_mixer: &HyperConnectionWeights,
        token_count: i32,
    ) -> Result<MlxArray, TextExecutionError> {
        let lane_norm_scale = &self.weights.final_normalization_weight;
        let architecture =
            self.config
                .qwen4_exp_architecture()
                .ok_or(TextExecutionError::InvalidInput {
                    description: "only a Flash-Next model mixes residual lanes",
                })?;
        let lane_width = lanes.shape()[2];
        let last_lanes = self.runtime.slice(
            lanes,
            &[0, token_count - 1, 0],
            &[1, token_count, lane_width],
            &[1, 1, 1],
        )?;
        let (mixed, _) =
            self.read_lanes(&last_lanes, lane_norm_scale, lane_mixer, &architecture)?;
        let logits = self.quantized_linear(&mixed, &self.weights.language_model_head_weights)?;
        Ok(self
            .runtime
            .astype(&logits, mlx_bridge::MlxDtype::Float32)?)
    }

    /// Float32 logits for every row of `lanes`, `[1, rows, vocabulary]`: what
    /// verifying a drafted block compares each draft against.
    pub fn lane_mixed_vocabulary_logits_for_every_row(
        &self,
        lanes: &MlxArray,
    ) -> Result<MlxArray, TextExecutionError> {
        let (Some(lane_mixer), Some(architecture)) = (
            self.weights.final_lane_mixer.as_deref(),
            self.config.qwen4_exp_architecture(),
        ) else {
            return Err(TextExecutionError::InvalidInput {
                description: "only a Flash-Next model mixes residual lanes",
            });
        };
        let (mixed, _) = self.read_lanes(
            lanes,
            &self.weights.final_normalization_weight,
            lane_mixer,
            &architecture,
        )?;
        let logits = self.quantized_linear(&mixed, &self.weights.language_model_head_weights)?;
        Ok(self
            .runtime
            .astype(&logits, mlx_bridge::MlxDtype::Float32)?)
    }

    /// What the first decoder layer reads: the embeddings themselves, or for
    /// Flash-Next the residual lanes they start. Embeddings already widened
    /// into lanes pass through, so every forward entry can call this.
    pub fn decoder_layer_input(
        &self,
        embeddings: MlxArray,
    ) -> Result<MlxArray, TextExecutionError> {
        let Some(architecture) = self.config.qwen4_exp_architecture() else {
            return Ok(embeddings);
        };
        if embeddings.shape().last().copied() != Some(self.config.hidden_size() as i32) {
            return Ok(embeddings);
        }
        self.lanes_from_embeddings(&embeddings, &architecture)
    }

    /// The first lanes of a forward: every lane starts as the embedding.
    pub fn lanes_from_embeddings(
        &self,
        embeddings: &MlxArray,
        architecture: &Qwen4ExpArchitecture,
    ) -> Result<MlxArray, TextExecutionError> {
        let lane_count = architecture.residual_lane_count as usize;
        let copies = vec![embeddings; lane_count];
        Ok(self.runtime.concatenate_axis(&copies, -1)?)
    }
}

/// `[.., lanes * hidden]` viewed as `[.., lanes, hidden]`.
fn lane_split_shape(shape: &[i32], lane_count: i32, hidden_size: i32) -> Vec<i32> {
    let mut lane_shape = shape[..shape.len() - 1].to_vec();
    lane_shape.extend([lane_count, hidden_size]);
    lane_shape
}

#[cfg(test)]
#[path = "hyper_connection_tests.rs"]
mod hyper_connection_tests;
