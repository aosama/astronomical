//! Geometry contracts for `qwen4_exp` request state.
//!
//! The user-visible outcome under test: a request's four state streams are
//! named, laid out, and measured from validated configuration alone, so
//! admission projects real bytes and the utilization split can attribute
//! them. Every expectation is derived from the configuration in the test,
//! never from one artifact's numbers.

use astronomical_model_serving::{
    DecoderCacheLayerLayout, DecoderCacheTensorDtype, Qwen4ExpConfig, Qwen4ExpConfigError,
    Qwen4ExpDecoderLayerCacheDtypes, Qwen4ExpStateGeometry, qwen4_exp_decoder_cache_layout,
};

fn family_document_json(layer_count: usize, interval: usize) -> String {
    let mut layer_types = Vec::with_capacity(layer_count);
    for index in 0..layer_count {
        if (index + 1) % interval == 0 {
            layer_types.push("\"full_attention\"");
        } else {
            layer_types.push("\"linear_attention\"");
        }
    }
    format!(
        r#"{{
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "model_type": "qwen4_exp",
        "text_config": {{
            "model_type": "qwen4_exp_text",
            "hidden_size": 64,
            "num_hidden_layers": {layer_count},
            "layer_types": [{}],
            "full_attention_interval": {interval},
            "head_dim": 8,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "vocab_size": 128,
            "max_position_embeddings": 4096,
            "rms_norm_eps": 1e-06,
            "dtype": "bfloat16",
            "eos_token_id": 7,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_key_head_dim": 4,
            "linear_value_head_dim": 4,
            "linear_conv_kernel_dim": 4,
            "mamba_ssm_dtype": "float32",
            "indexer_budget": 64,
            "indexer_compress_ratio": 4,
            "indexer_head_dim": 4,
            "indexer_kv_heads": 1,
            "indexer_n_heads": 2,
            "hc_count": 2,
            "hc_lowrank": 8,
            "ngram_size": 3,
            "heads_per_ngram": 2,
            "ngram_vocab_size_base": 1000,
            "split_ngram_parts": 4,
            "ple_embed_dim": 64,
            "ple_layer_ids": [2],
            "ple_conv_kernel_size": 4,
            "make_ngram_vocab_size_divisible_by": 8
        }}
    }}"#,
        layer_types.join(", ")
    )
}

fn family_config(layer_count: usize, interval: usize) -> Qwen4ExpConfig {
    Qwen4ExpConfig::from_json_bytes(family_document_json(layer_count, interval).as_bytes())
        .expect("a compliant family document validates")
}

fn layer_dtypes(config: &Qwen4ExpConfig) -> Vec<Qwen4ExpDecoderLayerCacheDtypes> {
    config
        .layer_schedule
        .iter()
        .map(|kind| match kind {
            astronomical_model_serving::Qwen4ExpLayerKind::LinearAttention => {
                Qwen4ExpDecoderLayerCacheDtypes::LinearAttention {
                    convolution: DecoderCacheTensorDtype::BFloat16,
                }
            }
            astronomical_model_serving::Qwen4ExpLayerKind::FullAttention => {
                Qwen4ExpDecoderLayerCacheDtypes::FullAttention {
                    keys: DecoderCacheTensorDtype::BFloat16,
                    values: DecoderCacheTensorDtype::BFloat16,
                    index_keys: DecoderCacheTensorDtype::BFloat16,
                    index_values: DecoderCacheTensorDtype::BFloat16,
                }
            }
        })
        .collect()
}

#[test]
fn should_build_a_layout_with_one_entry_per_declared_layer() {
    let config = family_config(8, 4);
    let layout = qwen4_exp_decoder_cache_layout(&config, 256, &layer_dtypes(&config))
        .expect("a compliant configuration builds a layout");
    assert_eq!(layout.layer_count(), 8);
    for (index, kind) in config.layer_schedule.iter().enumerate() {
        let layer = layout.layer(index).expect("every layer has a layout");
        match (kind, layer) {
            (
                astronomical_model_serving::Qwen4ExpLayerKind::FullAttention,
                DecoderCacheLayerLayout::Composite { components },
            ) => {
                assert_eq!(
                    components.len(),
                    2,
                    "a full-attention layer carries the main key-value state plus the index cache"
                );
            }
            (
                astronomical_model_serving::Qwen4ExpLayerKind::LinearAttention,
                DecoderCacheLayerLayout::Composite { components },
            ) => {
                assert_eq!(
                    components.len(),
                    2,
                    "a linear-attention layer carries convolution plus recurrent state"
                );
            }
            _ => panic!("layer {index} has an unexpected layout"),
        }
    }
}

#[test]
fn should_measure_per_stream_bytes_from_configuration_alone() {
    let config = family_config(8, 4);
    let geometry = Qwen4ExpStateGeometry::measure(&config, &layer_dtypes(&config), 256)
        .expect("a compliant configuration measures");
    // Six linear-attention layers: convolution fixed state is
    // (kernel - 1) x (key heads * key dim + value heads * value dim) = 3 x 24
    // bf16 elements = 144 bytes; the recurrent accumulator is
    // value_heads x value_dim x key_dim float32 = 4 x 4 x 4 x 4 = 256 bytes.
    // Per layer 400 bytes, across six layers 2,400.
    assert_eq!(geometry.linear_attention_layers, 6);
    assert_eq!(geometry.linear_attention_fixed_bytes, 6 * 400);
    // Two full-attention layers: main key-value state is
    // 2 (kv heads) x 8 (head dim) x 2 (keys + values) x 2 (bf16) = 64
    // bytes per token per layer.
    assert_eq!(geometry.full_attention_layers, 2);
    assert_eq!(geometry.full_attention_bytes_per_token, 2 * 64);
    // The index cache is 1 (index kv head) x 4 (index head dim) x 2 x 2
    // = 16 bytes per token per layer.
    assert_eq!(geometry.index_key_cache_bytes_per_token, 2 * 16);
    // Hyper-connection streams: 8 layers x 2 sites x 2 streams x 64 wide
    // x 2 bytes = 4,096 transient bytes per forward.
    assert_eq!(geometry.hyper_connection_stream_workspace_bytes, 4_096);
    assert_eq!(geometry.persisted_bytes_per_token(), 2 * (64 + 16));
}

#[test]
fn should_project_totals_for_zero_and_single_token_requests() {
    let config = family_config(8, 4);
    let geometry = Qwen4ExpStateGeometry::measure(&config, &layer_dtypes(&config), 256)
        .expect("a compliant configuration measures");
    assert_eq!(
        geometry.persisted_bytes_at(0),
        geometry.linear_attention_fixed_bytes,
        "a zero-token request pays only the fixed linear-attention state"
    );
    assert_eq!(
        geometry.persisted_bytes_at(1),
        geometry.linear_attention_fixed_bytes + geometry.persisted_bytes_per_token(),
    );
}

#[test]
fn should_scale_persisted_state_linearly_with_tokens() {
    let config = family_config(8, 4);
    let geometry = Qwen4ExpStateGeometry::measure(&config, &layer_dtypes(&config), 256)
        .expect("a compliant configuration measures");
    let at_1k = geometry.persisted_bytes_at(1_000);
    let at_2k = geometry.persisted_bytes_at(2_000);
    assert_eq!(
        at_2k - at_1k,
        geometry.persisted_bytes_per_token() * 1_000,
        "persisted growth is exactly linear in tokens"
    );
}

#[test]
fn should_reject_a_dtype_slice_that_does_not_cover_every_layer() {
    let config = family_config(8, 4);
    let mut dtypes = layer_dtypes(&config);
    dtypes.pop();
    let error = qwen4_exp_decoder_cache_layout(&config, 256, &dtypes)
        .expect_err("a short dtype slice must fail");
    assert!(
        error
            .to_string()
            .contains("count 7 differs from model layer count 8"),
        "the error should name both counts: {error}"
    );
}

#[test]
fn should_reject_a_configuration_without_the_state_geometry_groups() {
    let mut document = family_document_json(8, 4);
    // Removing one field of the two-field group is a partial declaration.
    document = document.replace("\"hc_lowrank\": 8,", "");
    let error = Qwen4ExpConfig::from_json_bytes(document.as_bytes())
        .expect_err("a partial hyper-connection group must fail");
    assert_eq!(
        error,
        Qwen4ExpConfigError::PartialFieldGroup {
            group: "hyper-connection"
        }
    );
}
