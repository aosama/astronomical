use std::collections::BTreeSet;

use astronomical_model_serving::{
    Qwen3_5Config, Qwen3_5FeedForwardArchitecture, qwen3_5_language_tensor_profiles,
};
use serde_json::json;

use crate::common::qwen3_5::certified_dense_qwen3_6_config;
use crate::qwen3_5_hermetic::config::support::minimal_valid_config_json;

#[test]
fn should_classify_the_dense_model_type_as_dense_feed_forward_architecture() {
    let dense_config = certified_dense_qwen3_6_config();

    assert_eq!(
        dense_config.feed_forward_architecture(),
        Qwen3_5FeedForwardArchitecture::Dense
    );
}

#[test]
fn should_not_add_sparse_router_gate_profiles_when_resolving_a_dense_model() {
    let mut dense_config_document = minimal_valid_config_json();
    dense_config_document["architectures"] = json!(["Qwen3_5ForConditionalGeneration"]);
    dense_config_document["model_type"] = json!("qwen3_5");
    dense_config_document["text_config"]["model_type"] = json!("qwen3_5_text");
    dense_config_document["text_config"]["num_experts"] = json!(0);
    dense_config_document["text_config"]["num_experts_per_tok"] = json!(0);
    dense_config_document["text_config"]["moe_intermediate_size"] = json!(0);
    dense_config_document["text_config"]["shared_expert_intermediate_size"] = json!(0);
    dense_config_document["text_config"]["intermediate_size"] = json!(512);
    let dense_config_bytes = serde_json::to_vec(&dense_config_document)
        .expect("the dense test config should serialize as JSON");
    let mut dense_config = Qwen3_5Config::from_json_bytes(&dense_config_bytes)
        .expect("the dense test config should parse");

    dense_config.resolve_unquantized_modules_from_shard_index(&BTreeSet::new());

    assert!(
        !dense_config
            .quantized_module_profiles()
            .contains_key("language_model.model.layers.0.mlp.gate")
    );
}

#[test]
fn should_use_tied_embedding_weights_without_requiring_a_separate_language_model_head() {
    let mut dense_config_document = minimal_valid_config_json();
    dense_config_document["architectures"] = json!(["Qwen3_5ForConditionalGeneration"]);
    dense_config_document["model_type"] = json!("qwen3_5");
    dense_config_document["tie_word_embeddings"] = json!(true);
    dense_config_document["text_config"]["model_type"] = json!("qwen3_5_text");
    dense_config_document["text_config"]["num_experts"] = json!(0);
    dense_config_document["text_config"]["num_experts_per_tok"] = json!(0);
    dense_config_document["text_config"]["moe_intermediate_size"] = json!(0);
    dense_config_document["text_config"]["shared_expert_intermediate_size"] = json!(0);
    dense_config_document["text_config"]["intermediate_size"] = json!(512);
    let dense_config_bytes = serde_json::to_vec(&dense_config_document)
        .expect("the tied-embedding test config should serialize as JSON");

    let dense_config = Qwen3_5Config::from_json_bytes(&dense_config_bytes)
        .expect("a tied-embedding dense config should parse");
    let language_tensor_names = qwen3_5_language_tensor_profiles(&dense_config)
        .into_iter()
        .map(|tensor_profile| tensor_profile.name)
        .collect::<BTreeSet<_>>();

    assert!(dense_config.has_tied_embeddings());
    assert!(language_tensor_names.contains("language_model.model.embed_tokens.weight"));
    assert!(!language_tensor_names.contains("language_model.lm_head.weight"));
}
