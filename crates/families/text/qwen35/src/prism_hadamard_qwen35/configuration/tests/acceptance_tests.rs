//! Acceptance contracts: the shipped document shape parses into the validated
//! configuration with every dimension and module preserved.

use super::{parse_config, shipped_shape_config_json};

#[test]
fn should_parse_the_shipped_document_shape() {
    let config =
        parse_config(&shipped_shape_config_json()).expect("the fixture config should parse");
    let text = config.text();
    assert_eq!(text.hidden_size, 64);
    assert_eq!(text.intermediate_size, 128);
    assert_eq!(text.num_hidden_layers, 4);
    assert_eq!(text.num_attention_heads, 4);
    assert_eq!(text.num_key_value_heads, 2);
    assert_eq!(text.head_dim, 16);
    assert_eq!(text.vocab_size, 256);
    assert_eq!(text.full_attention_interval, 4);
    assert_eq!(text.linear_num_value_heads, 8);
    assert_eq!(text.linear_num_key_heads, 4);
    assert_eq!(text.linear_key_head_dim, 16);
    assert_eq!(text.linear_value_head_dim, 16);
    assert_eq!(text.linear_conv_kernel_dim, 4);
    assert_eq!(text.max_position_embeddings, 4096);
    assert_eq!(text.eos_token_ids, vec![248044]);
    assert!(text.mrope_interleaved);
    assert_eq!(text.mrope_section, vec![5, 5, 6]);
    assert_eq!(
        text.layer_types,
        vec![
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ]
    );
    assert!(config.vision_declared());
}

#[test]
fn should_preserve_packed_module_records_in_declared_order() {
    let config =
        parse_config(&shipped_shape_config_json()).expect("the fixture config should parse");
    let modules = config.packed_modules();
    assert_eq!(modules.len(), 3);
    assert_eq!(modules[0].path(), "lm_head");
    assert!(!modules[0].is_embedding());
    assert_eq!(modules[1].path(), "model.embed_tokens");
    assert!(modules[1].is_embedding());
    assert_eq!(modules[2].path(), "model.layers.0.mlp.down_proj");
    for module in modules {
        assert_eq!(module.block(), 1024);
    }
    assert_eq!(modules[0].tensor_stem(), "language_model.lm_head");
    assert_eq!(
        modules[2].tensor_stem(),
        "language_model.model.layers.0.mlp.down_proj"
    );
}

#[test]
fn should_accept_an_array_eos_token_id() {
    let mut document = shipped_shape_config_json();
    document["text_config"]["eos_token_id"] = serde_json::json!([248044, 248046]);
    let config = parse_config(&document).expect("an array eos token id should parse");
    assert_eq!(config.text().eos_token_ids, vec![248044, 248046]);
}

#[test]
fn should_accept_a_vision_free_pack() {
    let mut document = shipped_shape_config_json();
    document["components"] = serde_json::json!({"text": true, "vision": false, "mtp": false});
    let config = parse_config(&document).expect("a text-only pack should parse");
    assert!(!config.vision_declared());
}

#[test]
fn should_accept_a_text_only_export_without_the_requires_runtime_marker() {
    // Real text-only exporter variants of schema version 2 omit the
    // requires-runtime marker and carry Transformers export metadata instead;
    // the family identity is pinned by the model type, tensor namespace, and
    // GDN layout, so the pack must load (the 0.2.79 Stable defect: such a
    // pack was refused with a bare config-validation failure).
    let mut document = shipped_shape_config_json();
    document
        .as_object_mut()
        .expect("the fixture is an object")
        .remove("requires_runtime");
    document["transformers_version"] = serde_json::json!("4.57.1");
    document["components"] = serde_json::json!({"text": true, "vision": false, "mtp": false});
    let config = parse_config(&document)
        .expect("a text-only export without the runtime marker should parse");
    assert!(!config.vision_declared());
    assert_eq!(config.packed_modules().len(), 3);
}
