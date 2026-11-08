use serde_json::{Value, json};

use super::classifies_transformer_index;

fn transformer_index(
    quantization_level: &str,
    double_stream_block_count: usize,
    single_stream_block_count: usize,
) -> Value {
    let mut weight_map = serde_json::Map::new();
    weight_map.insert(
        "time_guidance_embed.linear_1.weight".into(),
        json!("0.safetensors"),
    );
    for block_index in 0..double_stream_block_count {
        weight_map.insert(
            format!("transformer_blocks.{block_index}.attn.to_q.weight"),
            json!("0.safetensors"),
        );
    }
    for block_index in 0..single_stream_block_count {
        weight_map.insert(
            format!("single_transformer_blocks.{block_index}.attn.to_out.weight"),
            json!("1.safetensors"),
        );
    }
    json!({
        "metadata": {
            "quantization_level": quantization_level,
            "quantization_group_size": "64",
            "mflux_version": "0.17.5"
        },
        "weight_map": weight_map
    })
}

fn classifies(index: &Value) -> bool {
    classifies_transformer_index(index.to_string().as_bytes())
        .expect("a JSON index should classify without error")
}

#[test]
fn should_classify_the_klein_4b_mflux_transformer_index() {
    assert!(classifies(&transformer_index("4", 5, 20)));
}

#[test]
fn should_not_classify_klein_9b_or_a_partial_block_set() {
    assert!(!classifies(&transformer_index("4", 8, 24)));
    assert!(!classifies(&transformer_index("4", 5, 19)));
    assert!(!classifies(&transformer_index("4", 0, 0)));
}

#[test]
fn should_not_classify_another_quantization_level() {
    assert!(!classifies(&transformer_index("8", 5, 20)));
}

#[test]
fn should_not_classify_an_index_without_mflux_metadata() {
    let plain_index =
        json!({ "weight_map": { "transformer_blocks.0.attn.to_q.weight": "0.safetensors" } });
    assert!(!classifies(&plain_index));
    assert!(!classifies(&json!({})));
}

#[test]
fn should_report_bytes_that_are_not_json() {
    assert!(classifies_transformer_index(b"not json").is_err());
}
