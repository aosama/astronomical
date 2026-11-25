//! The `text_config` fields Qwen3.8-Flash-Next (`qwen4_exp`) adds to the
//! Qwen 3.5 layout, and the validated architecture built from them.
//!
//! Flash-Next keeps the Qwen 3.5 decoder stack (Gated DeltaNet and gated
//! full attention, routed experts with a shared expert) and adds three
//! pieces around it: a residual carried on several lanes mixed by
//! hyper-connections, a sparse-attention indexer on its full-attention
//! layers, and an n-gram lookup table added on one linear-attention layer.
//! The reference implementation builds the model by subclassing its Qwen 3.5
//! modules, which is why this family is served as a generation of the Qwen
//! 3.5 configuration rather than as a separate family (issue #2533).

use serde::Deserialize;

use super::config_validation::Qwen3_5ConfigError;

/// Wire fields present only on Flash-Next checkpoints. Every field is
/// optional on the wire so a Qwen 3.5 document parses with all of them
/// absent; [`Qwen4ExpTextFields::validated_architecture`] requires them for
/// a Flash-Next document.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq)]
pub(super) struct Qwen4ExpTextFields {
    #[serde(default)]
    hc_count: Option<u32>,
    #[serde(default)]
    hc_lowrank: Option<u32>,
    #[serde(default)]
    ple_layer_ids: Vec<u32>,
    #[serde(default)]
    ple_embed_dim: Option<u32>,
    #[serde(default)]
    ple_conv_kernel_size: Option<u32>,
    #[serde(default)]
    ngram_size: Option<u32>,
    #[serde(default)]
    heads_per_ngram: Option<u32>,
    #[serde(default)]
    split_ngram_parts: Option<u32>,
    #[serde(default)]
    ngram_vocab_size_base: Option<u64>,
    #[serde(default)]
    make_ngram_vocab_size_divisible_by: Option<u64>,
    #[serde(default)]
    indexer_n_heads: Option<u32>,
    #[serde(default)]
    indexer_kv_heads: Option<u32>,
    #[serde(default)]
    indexer_head_dim: Option<u32>,
    #[serde(default)]
    indexer_budget: Option<u32>,
    #[serde(default)]
    indexer_compress_ratio: Option<u32>,
    #[serde(default)]
    output_gate_type: Option<String>,
}

/// How the gated output norm of a linear-attention layer activates its gate.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum LinearAttentionOutputGate {
    /// Qwen 3.5: `rms_norm(output) * silu(gate)`.
    #[default]
    Silu,
    /// Flash-Next: `rms_norm(output) * sigmoid(gate)`.
    Sigmoid,
}

/// The validated Flash-Next additions every execution path reads.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Qwen4ExpArchitecture {
    /// Lanes the residual is carried on between blocks.
    pub residual_lane_count: u32,
    /// Width of the hyper-connection mixer's low-rank projection.
    pub hyper_connection_low_rank: u32,
    /// Zero-based index of the one decoder layer the n-gram table feeds.
    pub ngram_layer_index: u32,
    /// Width of one gathered n-gram embedding row set (all heads together).
    pub ngram_embedding_width: u32,
    pub ngram_convolution_kernel_size: u32,
    /// Longest n-gram hashed: 3 hashes pairs and triples.
    pub ngram_size: u32,
    pub ngram_heads_per_order: u32,
    pub ngram_table_shard_count: u32,
    /// Smallest hash table size an n-gram head may draw; each head's table
    /// is the next prime past the previous head's.
    pub ngram_vocabulary_size_base: u64,
    /// The concatenated tables are padded to a multiple of this.
    pub ngram_vocabulary_size_alignment: u64,
    pub indexer_query_head_count: u32,
    pub indexer_head_dimension: u32,
    /// Tokens of context the indexer keeps per query.
    pub indexer_token_budget: u32,
    /// Tokens averaged into one indexed key block.
    pub indexer_compress_ratio: u32,
    pub linear_attention_output_gate: LinearAttentionOutputGate,
    /// The token that resets n-gram history: the text configuration's own
    /// end-of-sequence id, which the reference hashes with, not the chat
    /// stop tokens generation ends on.
    pub ngram_end_of_sequence_token_id: u32,
}

impl Qwen4ExpArchitecture {
    /// Gathered rows per token: one per head of every n-gram order above one.
    #[must_use]
    pub const fn ngram_head_count(&self) -> u32 {
        (self.ngram_size - 1) * self.ngram_heads_per_order
    }

    /// Key blocks a query keeps once context exceeds the budget.
    #[must_use]
    pub const fn indexer_selected_block_count(&self) -> u32 {
        self.indexer_token_budget / self.indexer_compress_ratio
    }
}

impl Qwen4ExpTextFields {
    /// Builds the architecture, refusing a missing or unsupported field by
    /// name. `layer_is_linear_attention` answers for zero-based layers.
    pub(super) fn validated_architecture(
        &self,
        hidden_size: u32,
        layer_count: u32,
        end_of_sequence_token_id: Option<u32>,
        layer_is_linear_attention: impl Fn(usize) -> bool,
    ) -> Result<Qwen4ExpArchitecture, Qwen3_5ConfigError> {
        let mut architecture = self.residual_and_attention_settings()?;
        self.fill_ngram_settings(&mut architecture, hidden_size, layer_count)?;
        architecture.ngram_end_of_sequence_token_id = end_of_sequence_token_id.ok_or(invalid(
            "text_config.eos_token_id is required: the n-gram history resets on it",
        ))?;
        validate_shapes(&architecture, self.indexer_kv_heads)?;
        if !layer_is_linear_attention(architecture.ngram_layer_index as usize) {
            return Err(invalid(
                "text_config.ple_layer_ids must name a linear-attention layer",
            ));
        }
        Ok(architecture)
    }
}

impl Qwen4ExpTextFields {
    /// The lane, indexer and output-gate settings; n-gram settings default.
    fn residual_and_attention_settings(&self) -> Result<Qwen4ExpArchitecture, Qwen3_5ConfigError> {
        Ok(Qwen4ExpArchitecture {
            residual_lane_count: required("hc_count", self.hc_count)?,
            hyper_connection_low_rank: required("hc_lowrank", self.hc_lowrank)?,
            indexer_query_head_count: required("indexer_n_heads", self.indexer_n_heads)?,
            indexer_head_dimension: required("indexer_head_dim", self.indexer_head_dim)?,
            indexer_token_budget: required("indexer_budget", self.indexer_budget)?,
            indexer_compress_ratio: required(
                "indexer_compress_ratio",
                self.indexer_compress_ratio,
            )?,
            linear_attention_output_gate: output_gate(self.output_gate_type.as_deref())?,
            ..Qwen4ExpArchitecture::default()
        })
    }

    fn fill_ngram_settings(
        &self,
        architecture: &mut Qwen4ExpArchitecture,
        hidden_size: u32,
        layer_count: u32,
    ) -> Result<(), Qwen3_5ConfigError> {
        architecture.ngram_layer_index =
            single_ngram_layer_index(&self.ple_layer_ids, layer_count)?;
        architecture.ngram_embedding_width = self.ple_embed_dim.unwrap_or(hidden_size);
        architecture.ngram_convolution_kernel_size =
            required("ple_conv_kernel_size", self.ple_conv_kernel_size)?;
        architecture.ngram_size = required("ngram_size", self.ngram_size)?;
        architecture.ngram_heads_per_order = required("heads_per_ngram", self.heads_per_ngram)?;
        architecture.ngram_table_shard_count =
            required("split_ngram_parts", self.split_ngram_parts)?;
        architecture.ngram_vocabulary_size_base =
            required_wide("ngram_vocab_size_base", self.ngram_vocab_size_base)?;
        architecture.ngram_vocabulary_size_alignment = required_wide(
            "make_ngram_vocab_size_divisible_by",
            self.make_ngram_vocab_size_divisible_by,
        )?;
        Ok(())
    }
}

fn required(field_name: &str, value: Option<u32>) -> Result<u32, Qwen3_5ConfigError> {
    match value {
        Some(value) if value > 0 => Ok(value),
        _ => Err(Qwen3_5ConfigError::InvalidConfigValueDynamic {
            description: format!("text_config.{field_name} must be a positive integer"),
        }),
    }
}

fn required_wide(field_name: &str, value: Option<u64>) -> Result<u64, Qwen3_5ConfigError> {
    match value {
        Some(value) if value > 0 => Ok(value),
        _ => Err(Qwen3_5ConfigError::InvalidConfigValueDynamic {
            description: format!("text_config.{field_name} must be a positive integer"),
        }),
    }
}

fn invalid(description: &'static str) -> Qwen3_5ConfigError {
    Qwen3_5ConfigError::InvalidConfigValue { description }
}

/// The wire names one-based layers; exactly one n-gram layer is served.
fn single_ngram_layer_index(
    one_based_layer_ids: &[u32],
    layer_count: u32,
) -> Result<u32, Qwen3_5ConfigError> {
    match one_based_layer_ids {
        [one_based_layer_id] if (1..=layer_count).contains(one_based_layer_id) => {
            Ok(one_based_layer_id - 1)
        }
        _ => Err(invalid(
            "text_config.ple_layer_ids must name exactly one decoder layer",
        )),
    }
}

fn output_gate(
    output_gate_type: Option<&str>,
) -> Result<LinearAttentionOutputGate, Qwen3_5ConfigError> {
    match output_gate_type {
        Some("sigmoid") => Ok(LinearAttentionOutputGate::Sigmoid),
        Some("silu") => Ok(LinearAttentionOutputGate::Silu),
        _ => Err(invalid(
            "text_config.output_gate_type must be 'sigmoid' or 'silu'",
        )),
    }
}

fn validate_shapes(
    architecture: &Qwen4ExpArchitecture,
    indexer_key_head_count: Option<u32>,
) -> Result<(), Qwen3_5ConfigError> {
    if architecture.residual_lane_count < 2 {
        return Err(invalid("text_config.hc_count must be at least 2"));
    }
    if indexer_key_head_count != Some(1) {
        return Err(invalid("text_config.indexer_kv_heads must be 1"));
    }
    if architecture.ngram_size < 2 {
        return Err(invalid("text_config.ngram_size must be at least 2"));
    }
    if !architecture
        .indexer_token_budget
        .is_multiple_of(architecture.indexer_compress_ratio)
    {
        return Err(invalid(
            "text_config.indexer_budget must divide evenly by indexer_compress_ratio",
        ));
    }
    if !architecture
        .ngram_embedding_width
        .is_multiple_of(architecture.ngram_head_count())
    {
        return Err(invalid(
            "text_config.ple_embed_dim must divide evenly by the n-gram head count",
        ));
    }
    Ok(())
}

#[cfg(test)]
#[path = "qwen4_exp_fields_tests.rs"]
mod qwen4_exp_fields_tests;
