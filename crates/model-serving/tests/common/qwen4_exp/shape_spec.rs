//! Shape specification for generated `qwen4_exp` artifact variants.
//!
//! One declarative struct describes a complete compliant artifact: layer
//! schedule, subsystem geometry, per-projection quantization, lookup-table
//! form, and which sidecars exist. The writer in `writer.rs` materializes it
//! into a temporary directory, and the variant set in `variants.rs` names the
//! published packaging spread so every family test can run against a matrix
//! instead of one downloaded checkpoint.
//!
//! This is test-support code: nothing in production may reach it, so it
//! cannot become a hidden loading path.

/// How the n-gram lookup table is stored in a generated variant.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum NgramStorageForm {
    /// Inline shards named `shard_N.` with weight, scales, and biases.
    InlineUnderscoreNaming,
    /// Inline shards named `shards.N.` with weight, scales, and biases.
    InlineDotNaming,
    /// Inline shards named `shards.N.` with weight and scales only, matching
    /// the non-affine field set one published artifact uses.
    InlineDotNamingWithoutBiases,
    /// Inline shards plus a `ple-store.json` byte-range manifest, matching
    /// the artifact that externalizes addressing.
    InlineDotNamingWithManifest,
}

/// Quantization profile for one projection group.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VariantQuantization {
    /// No quantization document at all, matching the upstream-style variant.
    None,
    /// Affine at the given bits and group size.
    Affine { bits: u32, group_size: u32 },
    /// Block-scaled floating point at 4 bits, group 32.
    Mxfp4,
}

/// Which optional subsystems a variant declares.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VariantSubsystems {
    pub linear_attention: bool,
    pub sparse_attention: bool,
    pub hyper_connections: bool,
    pub multi_token_prediction_declared: bool,
}

/// Everything one generated variant needs, declared up front.
#[derive(Clone, Debug)]
pub struct Qwen4ExpShapeSpec {
    /// Fictional public identity; never a real published model name.
    pub model_id: &'static str,
    pub hidden_size: u32,
    pub decoder_layers: u32,
    pub full_attention_interval: u32,
    pub head_dim: u32,
    pub attention_heads: u32,
    pub key_value_heads: u32,
    pub vocabulary_size: u32,
    pub context_window_tokens: u64,
    pub eos_token_id: u32,
    /// Number of routed experts: 512 and 288 are the published pair.
    pub routed_experts: u32,
    pub expert_fan_out: u32,
    pub moe_intermediate_size: u32,
    /// Per-projection quantization; every tensor of the group shares it.
    pub quantization: VariantQuantization,
    /// Convolution weight axis order: published artifacts disagree.
    pub convolution_axes_channel_last: bool,
    /// Whether the indexer projection is quantized or left in BF16; both
    /// appear in published artifacts.
    pub indexer_projection_quantized: bool,
    /// Lookup-table storage form.
    pub ngram_storage: NgramStorageForm,
    pub ngram_shard_count: u32,
    /// Declares a prediction head without shipping tensors when false.
    pub multi_token_prediction_tensors: bool,
    pub subsystems: VariantSubsystems,
    /// Maximum bytes per shard file; the writer splits tensors across shards
    /// at this boundary, mirroring how published artifacts choose 954 MB or
    /// 5 GB shards.
    pub maximum_shard_bytes: usize,
}

impl Qwen4ExpShapeSpec {
    /// The layer schedule implied by the interval: the last layer of every
    /// interval block is a full-attention layer.
    #[must_use]
    pub fn layer_schedule(&self) -> Vec<bool> {
        (0..self.decoder_layers)
            .map(|index| {
                (u32::try_from(index).expect("layer index fits u32") + 1)
                    % self.full_attention_interval
                    == 0
            })
            .collect()
    }

    /// Packed unsigned 32-bit columns for a logical width at this variant's
    /// affine bits: 160, 240, or 320 for the published 2/3/4-bit profiles.
    #[must_use]
    pub fn packed_columns(&self, logical_width: u32) -> u32 {
        match self.quantization {
            VariantQuantization::None => logical_width,
            VariantQuantization::Affine { bits, .. } => logical_width * bits / 32,
            VariantQuantization::Mxfp4 => logical_width * 4 / 32,
        }
    }

    /// The per-head prime tables the lookup rule derives, computed here so a
    /// variant's row count is provably consistent with its configuration.
    ///
    /// # Errors
    /// When the derived head count cannot support the embedding width.
    pub fn ngram_layout(&self) -> Result<(Vec<u64>, Vec<u64>, u64), String> {
        let config = astronomical_model_serving::NgramIdentityConfiguration {
            ngram_size: 3,
            heads_per_ngram: 2,
            unigram_vocab_size: self.vocabulary_size,
            ngram_vocab_size_base: 1_000,
            vocabulary_divisor: 8,
            seed: 1234,
            ple_layer_ordinal: 0,
            eos_token_id: self.eos_token_id,
        };
        let identity = astronomical_model_serving::NgramRowIdentity::build(&config)
            .map_err(|error| error.to_string())?;
        let layout = identity.layout();
        let sizes: Vec<u64> = (0..layout.head_count())
            .map(|head| layout.head_size(head))
            .collect();
        let offsets: Vec<u64> = (0..layout.head_count())
            .map(|head| layout.head_offset(head))
            .collect();
        Ok((sizes, offsets, layout.padded_row_count()))
    }
}
