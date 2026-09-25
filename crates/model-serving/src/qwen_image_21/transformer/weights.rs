//! Qwen-Image-2.1 quantized linear loading: the 4-bit affine triple every projection uses.
//!
//! The reviewed artifact quantizes every transformer projection with the MLX affine scheme:
//! `weight` `[out, in/8]` uint32 (eight 4-bit values per word), and `scales`/`biases`
//! `[out, in/group_size]`. MLX consumes exactly this layout through
//! `quantized_matmul_affine` with `transpose_weights = true`, so the tensors load without any
//! repacking. The affine `biases` are dequantization offsets, not layer biases — the reference
//! model has no projection biases anywhere.

use astronomical_runtime_integration::{MlxArray, MlxSafetensors};

use crate::qwen_image_21::QwenImage21EngineError;
use crate::qwen_image_21::mlx_math::{QuantizedLinear, validate_shape};

use super::blocks::{
    QwenImage21AttentionWeights, QwenImage21BlockWeights, QwenImage21FeedForwardWeights,
};

/// Hidden width everywhere: `num_attention_heads × attention_head_dim`.
pub(super) const HIDDEN_WIDTH: usize = 4096;
/// Attention heads per layer (`num_attention_heads`).
pub(super) const HEAD_COUNT: usize = 32;
/// Channels per attention head (`attention_head_dim`).
pub(super) const HEAD_WIDTH: usize = 128;
/// Feed-forward hidden width: `hidden × mlp_ratio` (ratio 3).
pub(super) const FEED_FORWARD_WIDTH: usize = 12_288;
/// Latent channels entering (`img_in`) and leaving (`proj_out`) the transformer.
pub(super) const LATENT_CHANNEL_COUNT: usize = 64;
/// Width of the vision-language text embeddings entering `txt_in` — the Qwen3-VL encoder's
/// `hidden_size`, which this transformer's `context_in_dim` must match.
pub(super) const CONTEXT_INPUT_WIDTH: usize =
    crate::qwen_image_21::QWEN_IMAGE_21_TEXT_EMBEDDING_WIDTH;
/// The shared modulation projection emits four per-block tensors (`scale, gate` × 2).
pub(super) const MODULATION_WIDTH: usize = 4 * HIDDEN_WIDTH;
/// Sinusoidal timestep embedding width entering `time_text_embed.linear_1`.
pub(super) const TIMESTEP_EMBEDDING_WIDTH: usize = 256;

/// Every projection of the reviewed transformer, bound by weight name and validated shape.
#[derive(Debug)]
pub struct QwenImage21TransformerWeights {
    pub(super) img_in: QuantizedLinear,
    pub(super) text_norm: MlxArray,
    pub(super) txt_in_layer: QuantizedLinear,
    pub(super) txt_out_layer: QuantizedLinear,
    pub(super) time_embed_linear_1: QuantizedLinear,
    pub(super) time_embed_linear_2: QuantizedLinear,
    pub(super) modulation: QuantizedLinear,
    pub(super) blocks: Vec<QwenImage21BlockWeights>,
    pub(super) norm_out_linear: QuantizedLinear,
    pub(super) proj_out: QuantizedLinear,
}

impl QwenImage21TransformerWeights {
    pub fn load(
        tensors: &MlxSafetensors,
        block_count: usize,
    ) -> Result<Self, QwenImage21EngineError> {
        let text_norm = tensors.tensor("txt_in.text_norm.weight")?;
        validate_shape(
            "txt_in.text_norm",
            "weight",
            &text_norm,
            &[CONTEXT_INPUT_WIDTH],
        )?;
        let blocks = (0..block_count)
            .map(|block_index| load_block(tensors, block_index))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            img_in: QuantizedLinear::load(tensors, "img_in", LATENT_CHANNEL_COUNT, HIDDEN_WIDTH)?,
            text_norm,
            txt_in_layer: QuantizedLinear::load(
                tensors,
                "txt_in.in_layer",
                CONTEXT_INPUT_WIDTH,
                HIDDEN_WIDTH,
            )?,
            txt_out_layer: QuantizedLinear::load(
                tensors,
                "txt_in.out_layer",
                HIDDEN_WIDTH,
                HIDDEN_WIDTH,
            )?,
            time_embed_linear_1: QuantizedLinear::load(
                tensors,
                "time_text_embed.linear_1",
                TIMESTEP_EMBEDDING_WIDTH,
                HIDDEN_WIDTH,
            )?,
            time_embed_linear_2: QuantizedLinear::load(
                tensors,
                "time_text_embed.linear_2",
                HIDDEN_WIDTH,
                HIDDEN_WIDTH,
            )?,
            modulation: QuantizedLinear::load(
                tensors,
                "modulation.0",
                HIDDEN_WIDTH,
                MODULATION_WIDTH,
            )?,
            blocks,
            norm_out_linear: QuantizedLinear::load(
                tensors,
                "norm_out.linear",
                HIDDEN_WIDTH,
                HIDDEN_WIDTH,
            )?,
            proj_out: QuantizedLinear::load(
                tensors,
                "proj_out",
                HIDDEN_WIDTH,
                LATENT_CHANNEL_COUNT,
            )?,
        })
    }

    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }
}

fn load_block(
    tensors: &MlxSafetensors,
    block_index: usize,
) -> Result<QwenImage21BlockWeights, QwenImage21EngineError> {
    let prefix = format!("transformer_blocks.{block_index}");
    let attention = QwenImage21AttentionWeights {
        to_query: QuantizedLinear::load(
            tensors,
            &format!("{prefix}.attn.to_q"),
            HIDDEN_WIDTH,
            HIDDEN_WIDTH,
        )?,
        to_key: QuantizedLinear::load(
            tensors,
            &format!("{prefix}.attn.to_k"),
            HIDDEN_WIDTH,
            HIDDEN_WIDTH,
        )?,
        to_value: QuantizedLinear::load(
            tensors,
            &format!("{prefix}.attn.to_v"),
            HIDDEN_WIDTH,
            HIDDEN_WIDTH,
        )?,
        to_output: QuantizedLinear::load(
            tensors,
            &format!("{prefix}.attn.to_out.0"),
            HIDDEN_WIDTH,
            HIDDEN_WIDTH,
        )?,
        norm_query: tensors.tensor(&format!("{prefix}.attn.norm_q.weight"))?,
        norm_key: tensors.tensor(&format!("{prefix}.attn.norm_k.weight"))?,
    };
    validate_shape(
        &format!("{prefix}.attn"),
        "norm_q.weight",
        &attention.norm_query,
        &[HEAD_WIDTH],
    )?;
    validate_shape(
        &format!("{prefix}.attn"),
        "norm_k.weight",
        &attention.norm_key,
        &[HEAD_WIDTH],
    )?;
    let feed_forward = QwenImage21FeedForwardWeights {
        gate_layer: QuantizedLinear::load(
            tensors,
            &format!("{prefix}.img_mlp.gate_layer"),
            HIDDEN_WIDTH,
            FEED_FORWARD_WIDTH,
        )?,
        projection: QuantizedLinear::load(
            tensors,
            &format!("{prefix}.img_mlp.proj"),
            HIDDEN_WIDTH,
            FEED_FORWARD_WIDTH,
        )?,
        output: QuantizedLinear::load(
            tensors,
            &format!("{prefix}.img_mlp.out"),
            FEED_FORWARD_WIDTH,
            HIDDEN_WIDTH,
        )?,
    };
    Ok(QwenImage21BlockWeights {
        attention,
        feed_forward,
    })
}
