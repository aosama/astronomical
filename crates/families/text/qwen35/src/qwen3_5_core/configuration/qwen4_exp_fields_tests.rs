use serde_json::{Value, json};

use crate::qwen3_5_core::configuration::{
    LinearAttentionOutputGate, Qwen3_5Config, Qwen3_5ConfigError, Qwen3_5ModelGeneration,
};
use crate::qwen3_5_core::qwen4_exp_config_test_support::{
    frozen_flash_next_config, frozen_flash_next_config_document,
};

fn parse(document: &Value) -> Result<Qwen3_5Config, Qwen3_5ConfigError> {
    Qwen3_5Config::from_json_bytes(&document.to_string().into_bytes())
}

fn with_text_field(field_name: &str, value: Value) -> Value {
    let mut document = frozen_flash_next_config_document();
    document["text_config"][field_name] = value;
    document
}

#[test]
fn should_parse_the_published_flash_next_configuration_as_the_qwen4_exp_generation() {
    let config = frozen_flash_next_config();
    let architecture = config
        .qwen4_exp_architecture()
        .expect("a Flash-Next document carries its additions");
    assert_eq!(architecture.residual_lane_count, 4);
    assert_eq!(architecture.hyper_connection_low_rank, 320);
    // The wire names layers from one; layer 2 there is the second layer.
    assert_eq!(architecture.ngram_layer_index, 1);
    assert_eq!(architecture.ngram_head_count(), 16);
    assert_eq!(architecture.indexer_selected_block_count(), 512);
    assert_eq!(
        architecture.linear_attention_output_gate,
        LinearAttentionOutputGate::Sigmoid
    );
    assert_eq!(config.expert_count(), 512);
}

#[test]
fn should_keep_qwen3_5_documents_on_the_qwen3_5_generation() {
    let config = crate::qwen3_5_core::frozen_config_test_support::frozen_ornith_1_0_config();
    assert_eq!(config.model_generation(), Qwen3_5ModelGeneration::Qwen3_5);
    assert!(config.qwen4_exp_architecture().is_none());
}

#[test]
fn should_refuse_a_flash_next_document_without_its_lane_count() {
    let mut document = frozen_flash_next_config_document();
    document["text_config"]
        .as_object_mut()
        .expect("text_config is an object")
        .remove("hc_count");
    let parse_error = parse(&document).expect_err("a missing lane count must be refused");
    assert!(
        parse_error.to_string().contains("hc_count"),
        "{parse_error}"
    );
}

#[test]
fn should_refuse_a_single_residual_lane() {
    let parse_error = parse(&with_text_field("hc_count", json!(1)))
        .expect_err("one lane is not a multi-lane residual");
    assert!(
        parse_error.to_string().contains("hc_count"),
        "{parse_error}"
    );
}

#[test]
fn should_refuse_an_ngram_layer_on_a_full_attention_layer() {
    // One-based layer 4 is the first full-attention layer.
    let parse_error = parse(&with_text_field("ple_layer_ids", json!([4])))
        .expect_err("the n-gram table feeds only a linear-attention layer");
    assert!(
        parse_error.to_string().contains("ple_layer_ids"),
        "{parse_error}"
    );
}

#[test]
fn should_refuse_an_ngram_layer_past_the_last_layer() {
    let parse_error = parse(&with_text_field("ple_layer_ids", json!([49])))
        .expect_err("layer 49 does not exist in a 48-layer stack");
    assert!(
        parse_error.to_string().contains("ple_layer_ids"),
        "{parse_error}"
    );
}

#[test]
fn should_refuse_more_than_one_indexer_key_head() {
    let parse_error = parse(&with_text_field("indexer_kv_heads", json!(2)))
        .expect_err("the indexer scores against one key head");
    assert!(
        parse_error.to_string().contains("indexer_kv_heads"),
        "{parse_error}"
    );
}

#[test]
fn should_refuse_a_budget_that_does_not_divide_into_blocks() {
    let parse_error = parse(&with_text_field("indexer_budget", json!(2050)))
        .expect_err("a budget must be whole blocks");
    assert!(
        parse_error.to_string().contains("indexer_budget"),
        "{parse_error}"
    );
}

#[test]
fn should_refuse_an_unknown_output_gate() {
    let parse_error = parse(&with_text_field("output_gate_type", json!("tanh")))
        .expect_err("only sigmoid and silu gates exist");
    assert!(
        parse_error.to_string().contains("output_gate_type"),
        "{parse_error}"
    );
}

#[test]
fn should_refuse_a_qwen4_exp_model_type_with_a_qwen3_5_text_model_type() {
    let parse_error = parse(&with_text_field("model_type", json!("qwen3_5_moe_text")))
        .expect_err("the text model type must match the generation");
    assert!(
        parse_error.to_string().contains("model_type"),
        "{parse_error}"
    );
}
