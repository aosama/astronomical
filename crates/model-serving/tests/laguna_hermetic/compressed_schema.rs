use astronomical_model_serving::{
    LagunaGlobalTensorRole, LagunaLayerTensorRole, LagunaNormalizationError,
    LagunaTargetNormalizer, LagunaTensorComponent, LagunaTensorId, LagunaTensorStorageEncoding,
};
use serde_json::json;

use super::compressed_artifact_support::{
    PUBLISHED_M1_NVFP4_QUANTIZATION_CONFIG, published_m1_nvfp4_dense_fixture,
    published_m1_nvfp4_sparse_fixture,
};
use super::support::{config_bytes, config_value};

#[test]
fn should_apply_published_nvfp4_only_to_selected_feed_forward_modules() {
    let model_directory = tempfile::tempdir().expect("the test should create a directory");
    published_m1_nvfp4_dense_fixture().write(model_directory.path());

    let artifact = astronomical_model_serving::LagunaArtifactValidator::new()
        .validate(model_directory.path())
        .expect("the selected-module NVFP4 artifact should validate");
    let dense_gate = artifact
        .tensor_contract()
        .descriptor(&layer_weight(LagunaLayerTensorRole::DenseFeedForward(
            astronomical_model_serving::LagunaExpertProjection::Gate,
        )))
        .expect("the dense gate should exist");
    let attention_query = artifact
        .tensor_contract()
        .descriptor(&layer_weight(LagunaLayerTensorRole::Attention(
            astronomical_model_serving::LagunaAttentionProjection::Query,
        )))
        .expect("the attention query should exist");
    let embedding = artifact
        .tensor_contract()
        .descriptor(&LagunaTensorId::Global {
            role: LagunaGlobalTensorRole::TokenEmbedding,
            component: LagunaTensorComponent::Weight,
        })
        .expect("the embedding should exist");
    let output_head = artifact
        .tensor_contract()
        .descriptor(&LagunaTensorId::Global {
            role: LagunaGlobalTensorRole::OutputHead,
            component: LagunaTensorComponent::Weight,
        })
        .expect("the output head should exist");

    assert!(matches!(
        dense_gate.storage_encoding(),
        LagunaTensorStorageEncoding::TwoLevelCompressedNvfp4 { .. }
    ));
    assert_eq!(
        attention_query.storage_encoding(),
        &LagunaTensorStorageEncoding::Unquantized
    );
    assert_eq!(
        embedding.storage_encoding(),
        &LagunaTensorStorageEncoding::Unquantized
    );
    assert_eq!(
        output_head.storage_encoding(),
        &LagunaTensorStorageEncoding::Unquantized
    );

    let sparse_directory = tempfile::tempdir().expect("the test should create a directory");
    published_m1_nvfp4_sparse_fixture().write(sparse_directory.path());
    let sparse_artifact = astronomical_model_serving::LagunaArtifactValidator::new()
        .validate(sparse_directory.path())
        .expect("selected sparse experts should validate");
    let router = sparse_artifact
        .tensor_contract()
        .descriptor(&layer_weight(LagunaLayerTensorRole::Router))
        .expect("the router should exist");
    assert_eq!(
        router.storage_encoding(),
        &LagunaTensorStorageEncoding::Unquantized
    );
}

#[test]
fn should_reject_unsupported_or_contradictory_compressed_semantics() {
    let published: serde_json::Value = serde_json::from_str(PUBLISHED_M1_NVFP4_QUANTIZATION_CONFIG)
        .expect("the public quantization_config should decode");
    let mut cases = Vec::new();

    let mut unknown_selector = published.clone();
    unknown_selector["config_groups"]["group_0"]["targets"] = json!(["re:.*future_proj$"]);
    cases.push(unknown_selector);
    let mut asymmetric = published.clone();
    asymmetric["config_groups"]["group_0"]["weights"]["symmetric"] = json!(false);
    cases.push(asymmetric);
    let mut unsupported_activation = published.clone();
    unsupported_activation["config_groups"]["group_0"]["input_activations"]["dynamic"] =
        json!(true);
    cases.push(unsupported_activation);
    let mut unsupported_kv_cache = published.clone();
    unsupported_kv_cache["kv_cache_scheme"]["strategy"] = json!("group");
    cases.push(unsupported_kv_cache);
    let mut unsupported_ignore = published;
    unsupported_ignore["ignore"] = json!(["re:.*\\.mlp\\.gate_proj$"]);
    cases.push(unsupported_ignore);

    for quantization_config in cases {
        let mut config = config_value(1);
        config["quantization_config"] = quantization_config;
        assert!(matches!(
            LagunaTargetNormalizer::normalize(&config_bytes(&config)),
            Err(LagunaNormalizationError::UnsupportedQuantizationValue { .. })
                | Err(LagunaNormalizationError::ConflictingQuantizationDocuments)
        ));
    }
}

fn layer_weight(role: LagunaLayerTensorRole) -> LagunaTensorId {
    LagunaTensorId::Layer {
        layer_index: 0,
        role,
        component: LagunaTensorComponent::Weight,
    }
}
