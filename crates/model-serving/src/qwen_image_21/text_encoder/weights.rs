//! Weight binding for the Qwen3-VL text encoder (text-only path).
//!
//! Every projection is the artifact's 4-bit affine quantization — the same `[out, in/8]` layout
//! the denoising transformer uses, bound through the family's shared `QuantizedLinear`. The
//! token embedding is the one different layout: the table is stored `[vocab, hidden/8]` (packed
//! along the *output*), which `quantized_matmul_affine` consumes untransposed against a one-hot
//! activation — an exact row dequantization without materializing the 1.2 GB dense table.
//! `lm_head` is deliberately absent: the encoder use only needs hidden states, never logits.
//! The vision tower stays absent for the same reason — text-only prompts never touch it.

use astronomical_runtime_integration::{MlxArray, MlxDtype, MlxRuntime, MlxSafetensors};

use crate::qwen_image_21::QwenImage21EngineError;
use crate::qwen_image_21::mlx_math::{
    QUANT_BITS, QUANT_GROUP_SIZE, QUANT_VALUES_PER_WORD, QuantizedLinear, validate_shape,
};

/// The artifact path every embedding tensor hangs off.
const EMBEDDING_PREFIX: &str = "language_model.model.embed_tokens";

/// The reviewed artifact's language-model geometry. The hidden width is the Qwen3-VL
/// `hidden_size`, which is also the width of the embeddings the denoising transformer consumes.
pub(super) const HIDDEN_WIDTH: usize = crate::qwen_image_21::QWEN_IMAGE_21_TEXT_EMBEDDING_WIDTH;
pub(super) const LAYER_COUNT: usize = 36;
pub(super) const QUERY_HEAD_COUNT: usize = 32;
pub(super) const KEY_VALUE_HEAD_COUNT: usize = 8;
pub(super) const HEAD_WIDTH: usize = 128;
pub(super) const FEED_FORWARD_WIDTH: usize = 12_288;
pub(super) const VOCAB_SIZE: usize = 151_936;

#[derive(Debug)]
pub(super) struct QuantizedEmbedding {
    weight: MlxArray,
    scales: MlxArray,
    biases: MlxArray,
}

impl QuantizedEmbedding {
    pub(super) fn load(tensors: &MlxSafetensors) -> Result<Self, QwenImage21EngineError> {
        let weight = tensors.tensor("language_model.model.embed_tokens.weight")?;
        let scales = tensors.tensor("language_model.model.embed_tokens.scales")?;
        let biases = tensors.tensor("language_model.model.embed_tokens.biases")?;
        validate_shape(
            EMBEDDING_PREFIX,
            "weight",
            &weight,
            &[VOCAB_SIZE, HIDDEN_WIDTH / QUANT_VALUES_PER_WORD],
        )?;
        validate_shape(
            EMBEDDING_PREFIX,
            "scales",
            &scales,
            &[VOCAB_SIZE, HIDDEN_WIDTH / QUANT_GROUP_SIZE],
        )?;
        validate_shape(
            EMBEDDING_PREFIX,
            "biases",
            &biases,
            &[VOCAB_SIZE, HIDDEN_WIDTH / QUANT_GROUP_SIZE],
        )?;
        Ok(Self {
            weight,
            scales,
            biases,
        })
    }

    /// Looks up `token_ids` rows: one-hot times the quantized table, transposed layout.
    pub(super) fn forward(
        &self,
        runtime: &MlxRuntime,
        token_ids: &[u32],
    ) -> Result<MlxArray, QwenImage21EngineError> {
        let sequence_length = token_ids.len();
        if sequence_length == 0 {
            return Err(QwenImage21EngineError::InvalidInput {
                description: "the text encoder needs at least one token".to_owned(),
            });
        }
        // One-hot rows in the *dense vocabulary* axis; the quantized matmul dequantizes exactly
        // the selected table rows. Prompts are short, so the (tokens × vocab) one-hot is small.
        let mut one_hot = vec![0.0_f32; sequence_length * VOCAB_SIZE];
        for (token_position, &token_id) in token_ids.iter().enumerate() {
            let vocab_index = token_id as usize;
            if vocab_index >= VOCAB_SIZE {
                return Err(QwenImage21EngineError::InvalidInput {
                    description: format!("token id {token_id} exceeds the vocabulary"),
                });
            }
            one_hot[token_position * VOCAB_SIZE + vocab_index] = 1.0;
        }
        let one_hot_array =
            runtime.array_from_f32(&one_hot, &[sequence_length as i32, VOCAB_SIZE as i32])?;
        let typed_one_hot = runtime.astype(&one_hot_array, MlxDtype::BFloat16)?;
        let embedded = runtime.quantized_matmul_affine(
            &typed_one_hot,
            &self.weight,
            &self.scales,
            &self.biases,
            false,
            QUANT_GROUP_SIZE as i32,
            QUANT_BITS,
        )?;
        // The layers consume `(batch, tokens, hidden)`; the one-hot path has no batch axis.
        Ok(runtime.reshape(&embedded, &[1, sequence_length as i32, HIDDEN_WIDTH as i32])?)
    }
}

#[derive(Debug)]
pub(super) struct QwenImage21TextEncoderLayerWeights {
    pub(super) input_layernorm: MlxArray,
    pub(super) query: QuantizedLinear,
    pub(super) key: QuantizedLinear,
    pub(super) value: QuantizedLinear,
    pub(super) output: QuantizedLinear,
    pub(super) query_norm: MlxArray,
    pub(super) key_norm: MlxArray,
    pub(super) post_attention_layernorm: MlxArray,
    pub(super) gate: QuantizedLinear,
    pub(super) up: QuantizedLinear,
    pub(super) down: QuantizedLinear,
}

/// Every projection and norm of the reviewed text encoder, bound by weight name.
///
/// `language_model.model.norm` — the final RMSNorm — is deliberately absent: the reference
/// bypasses it (a forward hook returns the norm's input) because the transformer consumes the
/// pre-norm last-layer output.
#[derive(Debug)]
pub struct QwenImage21TextEncoderWeights {
    embedding: QuantizedEmbedding,
    layers: Vec<QwenImage21TextEncoderLayerWeights>,
}

impl QwenImage21TextEncoderWeights {
    pub fn load(
        tensors: &MlxSafetensors,
        layer_count: usize,
    ) -> Result<Self, QwenImage21EngineError> {
        let layers = (0..layer_count)
            .map(|layer_index| load_layer(tensors, layer_index))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            embedding: QuantizedEmbedding::load(tensors)?,
            layers,
        })
    }

    pub(super) fn embedding(&self) -> &QuantizedEmbedding {
        &self.embedding
    }

    pub(super) fn layers(&self) -> &[QwenImage21TextEncoderLayerWeights] {
        &self.layers
    }
}

fn load_layer(
    tensors: &MlxSafetensors,
    layer_index: usize,
) -> Result<QwenImage21TextEncoderLayerWeights, QwenImage21EngineError> {
    let prefix = format!("language_model.model.layers.{layer_index}");
    let layer = QwenImage21TextEncoderLayerWeights {
        input_layernorm: tensors.tensor(&format!("{prefix}.input_layernorm.weight"))?,
        query: QuantizedLinear::load(
            tensors,
            &format!("{prefix}.self_attn.q_proj"),
            HIDDEN_WIDTH,
            QUERY_HEAD_COUNT * HEAD_WIDTH,
        )?,
        key: QuantizedLinear::load(
            tensors,
            &format!("{prefix}.self_attn.k_proj"),
            HIDDEN_WIDTH,
            KEY_VALUE_HEAD_COUNT * HEAD_WIDTH,
        )?,
        value: QuantizedLinear::load(
            tensors,
            &format!("{prefix}.self_attn.v_proj"),
            HIDDEN_WIDTH,
            KEY_VALUE_HEAD_COUNT * HEAD_WIDTH,
        )?,
        output: QuantizedLinear::load(
            tensors,
            &format!("{prefix}.self_attn.o_proj"),
            QUERY_HEAD_COUNT * HEAD_WIDTH,
            HIDDEN_WIDTH,
        )?,
        query_norm: tensors.tensor(&format!("{prefix}.self_attn.q_norm.weight"))?,
        key_norm: tensors.tensor(&format!("{prefix}.self_attn.k_norm.weight"))?,
        post_attention_layernorm: tensors
            .tensor(&format!("{prefix}.post_attention_layernorm.weight"))?,
        gate: QuantizedLinear::load(
            tensors,
            &format!("{prefix}.mlp.gate_proj"),
            HIDDEN_WIDTH,
            FEED_FORWARD_WIDTH,
        )?,
        up: QuantizedLinear::load(
            tensors,
            &format!("{prefix}.mlp.up_proj"),
            HIDDEN_WIDTH,
            FEED_FORWARD_WIDTH,
        )?,
        down: QuantizedLinear::load(
            tensors,
            &format!("{prefix}.mlp.down_proj"),
            FEED_FORWARD_WIDTH,
            HIDDEN_WIDTH,
        )?,
    };
    validate_shape(
        &prefix,
        "input_layernorm.weight",
        &layer.input_layernorm,
        &[HIDDEN_WIDTH],
    )?;
    validate_shape(
        &prefix,
        "self_attn.q_norm.weight",
        &layer.query_norm,
        &[HEAD_WIDTH],
    )?;
    validate_shape(
        &prefix,
        "self_attn.k_norm.weight",
        &layer.key_norm,
        &[HEAD_WIDTH],
    )?;
    validate_shape(
        &prefix,
        "post_attention_layernorm.weight",
        &layer.post_attention_layernorm,
        &[HIDDEN_WIDTH],
    )?;
    Ok(layer)
}
