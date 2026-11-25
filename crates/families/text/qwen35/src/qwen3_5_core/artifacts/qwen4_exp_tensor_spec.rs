//! The language tensors a Flash-Next (`qwen4_exp`) checkpoint must ship.
//!
//! The decoder layers carry the Qwen 3.5 attention and expert tensors, but
//! no input or post-attention norm: each block instead reads the residual
//! lanes through a hyper-connection, whose own grouped norm replaces both.
//! The model ends in a hyper-connection mixer and has no final norm.
//! Full-attention layers add a sparse-attention indexer, and one
//! linear-attention layer adds the n-gram table.

use crate::qwen3_5_core::configuration::Qwen4ExpArchitecture;
use crate::qwen4_exp::ngram_table_geometry::NgramTableGeometry;
use crate::{TensorDtype, TensorProfile};

use super::tensor_spec::{
    append_qwen3_5_full_attention_tensor_profiles, append_qwen3_5_linear_attention_tensor_profiles,
    append_qwen3_5_moe_feed_forward_tensor_profiles,
    append_qwen3_5_quantized_affine_tensor_profiles, qwen3_5_tensor_profile,
};
use super::{Qwen3_5ArtifactError, Qwen3_5Config};

/// The tensor name stem every n-gram table shard shares.
pub const NGRAM_TABLE_SHARD_STEM: &str = "ple.ple_embedding.ngram_embedding.shards";

/// The complete Flash-Next language inventory, n-gram table shards included.
pub fn qwen4_exp_language_tensor_profiles(
    config: &Qwen3_5Config,
    architecture: &Qwen4ExpArchitecture,
) -> Result<Vec<TensorProfile>, Qwen3_5ArtifactError> {
    let hidden_size = config.hidden_size() as usize;
    let vocabulary_size = config.vocabulary_size() as usize;
    let mut profiles = Vec::new();
    append_qwen3_5_quantized_affine_tensor_profiles(
        &mut profiles,
        "language_model.model.embed_tokens",
        &[vocabulary_size],
        hidden_size,
        config.quantization_profile_for_module("language_model.model.embed_tokens"),
    )?;
    for layer_index in 0..config.layer_count() as usize {
        append_decoder_layer_profiles(&mut profiles, layer_index, config, architecture)?;
    }
    append_hyper_connection_profiles(
        &mut profiles,
        "language_model.model.hyper_connection_mixer",
        config,
        architecture,
        false,
    )?;
    append_qwen3_5_quantized_affine_tensor_profiles(
        &mut profiles,
        "language_model.lm_head",
        &[vocabulary_size],
        hidden_size,
        config.quantization_profile_for_module("language_model.lm_head"),
    )?;
    Ok(profiles)
}

/// Whether a tensor belongs to the n-gram table, which is read row by row
/// from storage instead of being held in memory.
#[must_use]
pub fn is_ngram_table_shard_tensor_name(tensor_name: &str) -> bool {
    tensor_name.contains(".ple.ple_embedding.ngram_embedding.shards.")
}

fn append_decoder_layer_profiles(
    profiles: &mut Vec<TensorProfile>,
    layer_index: usize,
    config: &Qwen3_5Config,
    architecture: &Qwen4ExpArchitecture,
) -> Result<(), Qwen3_5ArtifactError> {
    let hidden_size = config.hidden_size() as usize;
    let layer_prefix = format!("language_model.model.layers.{layer_index}");
    if config.decoder_layer_is_full_attention(layer_index) {
        append_qwen3_5_full_attention_tensor_profiles(
            profiles,
            &layer_prefix,
            hidden_size,
            config.query_head_count() as usize,
            config.key_value_head_count() as usize,
            config.head_dimension() as usize,
            config,
        )?;
        append_indexer_profiles(profiles, &layer_prefix, config, architecture)?;
    } else {
        append_qwen3_5_linear_attention_tensor_profiles(
            profiles,
            &layer_prefix,
            hidden_size,
            config,
        )?;
    }
    append_qwen3_5_moe_feed_forward_tensor_profiles(profiles, &layer_prefix, hidden_size, config)?;
    for connection_name in ["attn_hyper_connection", "mlp_hyper_connection"] {
        append_hyper_connection_profiles(
            profiles,
            &format!("{layer_prefix}.{connection_name}"),
            config,
            architecture,
            true,
        )?;
    }
    if layer_index == architecture.ngram_layer_index as usize {
        append_ngram_layer_profiles(profiles, &layer_prefix, config, architecture)?;
    }
    Ok(())
}

/// A hyper-connection reads every lane through one grouped norm, mixes them
/// down to one hidden state through a low-rank projection, and, inside a
/// decoder layer, weights how the block's output is written back to each
/// lane (`block_inject_weight`). The final mixer only mixes.
pub(super) fn append_hyper_connection_profiles(
    profiles: &mut Vec<TensorProfile>,
    connection_prefix: &str,
    config: &Qwen3_5Config,
    architecture: &Qwen4ExpArchitecture,
    writes_back_to_lanes: bool,
) -> Result<(), Qwen3_5ArtifactError> {
    let lane_width = config.hidden_size() as usize * architecture.residual_lane_count as usize;
    let low_rank = architecture.hyper_connection_low_rank as usize;
    profiles.push(qwen3_5_tensor_profile(
        format!("{connection_prefix}.hc_norm.weight"),
        TensorDtype::ModelFloat,
        vec![lane_width],
    ));
    let mut projections = vec![
        ("input_mix_weight_down", low_rank, lane_width),
        ("input_mix_weight_up", lane_width, low_rank),
    ];
    if writes_back_to_lanes {
        projections.push((
            "block_inject_weight",
            architecture.residual_lane_count as usize,
            lane_width,
        ));
    }
    for (projection_name, output_width, input_width) in projections {
        let module_name = format!("{connection_prefix}.{projection_name}");
        append_qwen3_5_quantized_affine_tensor_profiles(
            profiles,
            &module_name,
            &[output_width],
            input_width,
            config.quantization_profile_for_module(&module_name),
        )?;
    }
    Ok(())
}

pub(super) fn append_indexer_profiles(
    profiles: &mut Vec<TensorProfile>,
    layer_prefix: &str,
    config: &Qwen3_5Config,
    architecture: &Qwen4ExpArchitecture,
) -> Result<(), Qwen3_5ArtifactError> {
    let indexer_prefix = format!("{layer_prefix}.self_attn.indexer");
    let head_dimension = architecture.indexer_head_dimension as usize;
    // One key head beside the query heads, projected together.
    let projected_width = (architecture.indexer_query_head_count as usize + 1) * head_dimension;
    let module_name = format!("{indexer_prefix}.index_qk_proj");
    append_qwen3_5_quantized_affine_tensor_profiles(
        profiles,
        &module_name,
        &[projected_width],
        config.hidden_size() as usize,
        config.quantization_profile_for_module(&module_name),
    )?;
    for norm_name in ["q_layernorm", "k_layernorm"] {
        profiles.push(qwen3_5_tensor_profile(
            format!("{indexer_prefix}.{norm_name}.weight"),
            TensorDtype::ModelFloat,
            vec![head_dimension],
        ));
    }
    Ok(())
}

fn append_ngram_layer_profiles(
    profiles: &mut Vec<TensorProfile>,
    layer_prefix: &str,
    config: &Qwen3_5Config,
    architecture: &Qwen4ExpArchitecture,
) -> Result<(), Qwen3_5ArtifactError> {
    let hidden_size = config.hidden_size() as usize;
    let lane_width = hidden_size * architecture.residual_lane_count as usize;
    let embedding_width = architecture.ngram_embedding_width as usize;
    let ngram_prefix = format!("{layer_prefix}.ple");
    // Depthwise: one kernel per lane channel, in MLX's `[out, kernel, in]`.
    profiles.push(qwen3_5_tensor_profile(
        format!("{ngram_prefix}.conv1d.weight"),
        TensorDtype::ModelFloat,
        vec![
            lane_width,
            architecture.ngram_convolution_kernel_size as usize,
            1,
        ],
    ));
    for (projection_name, output_width) in [("key_proj", lane_width), ("value_proj", hidden_size)] {
        let module_name = format!("{ngram_prefix}.{projection_name}");
        append_qwen3_5_quantized_affine_tensor_profiles(
            profiles,
            &module_name,
            &[output_width],
            embedding_width,
            config.quantization_profile_for_module(&module_name),
        )?;
    }
    for norm_name in ["norm_conv", "norm_key", "norm_query"] {
        profiles.push(qwen3_5_tensor_profile(
            format!("{ngram_prefix}.{norm_name}.weight"),
            TensorDtype::ModelFloat,
            vec![lane_width],
        ));
    }
    append_ngram_table_profiles(profiles, &ngram_prefix, config, architecture)
}

fn append_ngram_table_profiles(
    profiles: &mut Vec<TensorProfile>,
    ngram_prefix: &str,
    config: &Qwen3_5Config,
    architecture: &Qwen4ExpArchitecture,
) -> Result<(), Qwen3_5ArtifactError> {
    let head_count = architecture.ngram_head_count() as usize;
    for (table_name, length) in [
        ("layer_multipliers", architecture.ngram_size as usize),
        ("ngram_heads_offsets", head_count),
        ("ngram_heads_vocab_sizes", head_count),
    ] {
        profiles.push(qwen3_5_tensor_profile(
            format!("{ngram_prefix}.ple_embedding.{table_name}"),
            TensorDtype::Int64,
            vec![length],
        ));
    }
    let geometry = NgramTableGeometry::for_architecture(architecture);
    let row_width = architecture.ngram_embedding_width as usize / head_count;
    let layer_prefix = ngram_prefix.trim_end_matches(".ple");
    for shard_index in 0..geometry.shard_count() {
        let module_name = format!("{layer_prefix}.{NGRAM_TABLE_SHARD_STEM}.{shard_index}");
        append_qwen3_5_quantized_affine_tensor_profiles(
            profiles,
            &module_name,
            &[geometry.shard_row_count(shard_index) as usize],
            row_width,
            config.quantization_profile_for_module(&module_name),
        )?;
    }
    Ok(())
}

#[cfg(test)]
#[path = "qwen4_exp_tensor_spec_tests.rs"]
mod qwen4_exp_tensor_spec_tests;
