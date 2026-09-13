//! The named variant set mirroring the published packaging spread.
//!
//! Every published axis of variation appears in at least one variant, so a
//! family implementation that hard-wires one artifact's packaging fails at
//! least one of them. Geometries are tiny — eight layers, a 128-token
//! vocabulary, kilobyte payloads — so the matrix runs inside routine bounds.

use super::shape_spec::{
    NgramStorageForm, Qwen4ExpShapeSpec, VariantQuantization, VariantSubsystems,
};

const FULL_SUBSYSTEMS: VariantSubsystems = VariantSubsystems {
    linear_attention: true,
    sparse_attention: true,
    hyper_connections: true,
    multi_token_prediction_declared: false,
};

fn base_spec(model_id: &'static str) -> Qwen4ExpShapeSpec {
    Qwen4ExpShapeSpec {
        model_id,
        hidden_size: 64,
        decoder_layers: 8,
        full_attention_interval: 4,
        head_dim: 8,
        attention_heads: 4,
        key_value_heads: 2,
        vocabulary_size: 128,
        context_window_tokens: 4096,
        eos_token_id: 7,
        routed_experts: 12,
        expert_fan_out: 4,
        moe_intermediate_size: 16,
        quantization: VariantQuantization::Affine {
            bits: 4,
            group_size: 64,
        },
        convolution_axes_channel_last: false,
        indexer_projection_quantized: true,
        ngram_storage: NgramStorageForm::InlineDotNaming,
        ngram_shard_count: 2,
        multi_token_prediction_tensors: false,
        subsystems: FULL_SUBSYSTEMS,
        maximum_shard_bytes: 64 * 1024,
    }
}

/// The named variant matrix. Field names state the axis each variant covers.
pub fn variant_matrix() -> Vec<Qwen4ExpShapeSpec> {
    let mut variants = Vec::new();
    // Baseline: complete subsystems, 4-bit group-64 affine, dot naming.
    variants.push(base_spec("variant-baseline"));
    // Pruned expert geometry: the published 288-of-512 spread, scaled down.
    let mut pruned = base_spec("variant-pruned-experts");
    pruned.routed_experts = 6;
    variants.push(pruned);
    // 2-bit affine: packed columns 160 for a 2560-wide projection upstream.
    let mut two_bit = base_spec("variant-2bit");
    two_bit.quantization = VariantQuantization::Affine {
        bits: 2,
        group_size: 32,
    };
    variants.push(two_bit);
    // 3-bit affine: packed columns 240 upstream.
    let mut three_bit = base_spec("variant-3bit");
    three_bit.quantization = VariantQuantization::Affine {
        bits: 3,
        group_size: 32,
    };
    variants.push(three_bit);
    // MXFP4: the non-affine published profile.
    let mut mxfp4 = base_spec("variant-mxfp4");
    mxfp4.quantization = VariantQuantization::Mxfp4;
    variants.push(mxfp4);
    // No quantization document at all: the upstream-style variant.
    let mut native = base_spec("variant-native");
    native.quantization = VariantQuantization::None;
    variants.push(native);
    // Underscore lookup naming: five published artifacts use it.
    let mut underscore = base_spec("variant-ngram-underscore-naming");
    underscore.ngram_storage = NgramStorageForm::InlineUnderscoreNaming;
    variants.push(underscore);
    // Weight-and-scales-only lookup fields: the non-affine field set.
    let mut no_biases = base_spec("variant-ngram-without-biases");
    no_biases.ngram_storage = NgramStorageForm::InlineDotNamingWithoutBiases;
    variants.push(no_biases);
    // Manifest-addressed lookup storage: one published artifact externalizes it.
    let mut manifest = base_spec("variant-ngram-manifest");
    manifest.ngram_storage = NgramStorageForm::InlineDotNamingWithManifest;
    variants.push(manifest);
    // Declares a prediction head without shipping tensors: one artifact does.
    let mut headless = base_spec("variant-mtp-declared-without-tensors");
    headless.subsystems.multi_token_prediction_declared = true;
    headless.multi_token_prediction_tensors = false;
    variants.push(headless);
    // Text-only: the optional subsystem groups are absent entirely.
    let mut text_only = base_spec("variant-text-only");
    text_only.subsystems = VariantSubsystems {
        linear_attention: false,
        sparse_attention: false,
        hyper_connections: false,
        multi_token_prediction_declared: false,
    };
    variants.push(text_only);
    // Convolution axis order flipped: published artifacts disagree on it.
    let mut channel_last = base_spec("variant-convolution-channel-last");
    channel_last.convolution_axes_channel_last = true;
    variants.push(channel_last);
    // Unquantized indexer projection: two published artifacts leave it BF16.
    let mut bf16_indexer = base_spec("variant-indexer-bf16");
    bf16_indexer.indexer_projection_quantized = false;
    variants.push(bf16_indexer);
    variants
}
