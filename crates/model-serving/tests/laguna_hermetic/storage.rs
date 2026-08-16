use astronomical_model_serving::{
    LagunaNormalizationError, LagunaStorageDescriptor, LagunaTargetNormalizer,
};
use serde_json::json;

use super::support::{config_bytes, config_value, normalize};

#[test]
fn should_normalize_unquantized_and_direct_affine_storage() {
    assert_eq!(
        normalize(config_value(2)).storage(),
        &LagunaStorageDescriptor::Unquantized
    );

    let mut affine_config = config_value(2);
    affine_config["quantization"] = json!({
        "bits": 4,
        "group_size": 64,
        "mode": "affine",
        "language_model.lm_head": {"bits": 8, "group_size": 32, "mode": "affine"},
        "model.embed_tokens": {"bits": 6, "group_size": 128, "mode": "affine"}
    });
    let contract = normalize(affine_config);
    let LagunaStorageDescriptor::DirectAffine(affine) = contract.storage() else {
        panic!("direct affine storage should be canonical");
    };
    assert_eq!(affine.default_profile().bits(), 4);
    assert_eq!(affine.default_profile().group_size(), 64);
    assert_eq!(affine.module_override_count(), 2);
    assert_eq!(affine.profile_for_module("lm_head").bits(), 8);
    assert_eq!(affine.profile_for_module("model.embed_tokens").bits(), 6);
}

#[test]
fn should_accept_every_mlx_affine_width_and_group_size() {
    for bit_width in [2, 3, 4, 5, 6, 8] {
        for group_size in [32, 64, 128] {
            let mut config = config_value(1);
            config["quantization"] = json!({
                "bits": bit_width,
                "group_size": group_size,
                "mode": "affine",
                "lm_head": {"bits": bit_width, "group_size": group_size}
            });
            let LagunaStorageDescriptor::DirectAffine(affine) = normalize(config).storage().clone()
            else {
                panic!("supported affine storage should normalize");
            };
            assert_eq!(affine.default_profile().bits(), bit_width);
            assert_eq!(affine.default_profile().group_size(), group_size);
        }
    }
}

#[test]
fn should_compare_quantization_copies_by_canonical_semantics() {
    let mut config = config_value(2);
    config["quantization"] = json!({
        "bits": 4,
        "group_size": 64,
        "mode": "affine",
        "lm_head": {"bits": 8, "group_size": 32, "mode": "affine"}
    });
    config["quantization_config"] = json!({
        "group_size": 64,
        "bits": 4,
        "language_model.lm_head": {"group_size": 32, "bits": 8}
    });

    let LagunaStorageDescriptor::DirectAffine(affine) = normalize(config).storage().clone() else {
        panic!("equivalent copies should produce affine storage");
    };
    assert_eq!(affine.module_override_count(), 1);
}

#[test]
fn should_reject_conflicting_quantization_copies_and_canonical_collisions() {
    let mut conflicting_copies = config_value(2);
    conflicting_copies["quantization"] = json!({"bits": 4, "group_size": 64});
    conflicting_copies["quantization_config"] = json!({"bits": 3, "group_size": 64});
    assert!(matches!(
        LagunaTargetNormalizer::normalize(&config_bytes(&conflicting_copies)),
        Err(LagunaNormalizationError::ConflictingQuantizationDocuments)
    ));

    let mut colliding_overrides = config_value(2);
    colliding_overrides["quantization"] = json!({
        "bits": 4,
        "group_size": 64,
        "lm_head": {"bits": 8, "group_size": 64},
        "language_model.lm_head": {"bits": 6, "group_size": 64}
    });
    assert!(matches!(
        LagunaTargetNormalizer::normalize(&config_bytes(&colliding_overrides)),
        Err(LagunaNormalizationError::ConflictingModuleOverride { .. })
    ));
}

#[test]
fn should_reject_unsupported_affine_values_and_compressed_encodings() {
    let unsupported_documents = [
        json!({"bits": 7, "group_size": 64}),
        json!({"bits": 4, "group_size": 16}),
        json!({"bits": 4, "group_size": 64, "mode": "nf4"}),
    ];
    for unsupported_document in unsupported_documents {
        let mut config = config_value(1);
        config["quantization"] = unsupported_document;
        assert!(matches!(
            LagunaTargetNormalizer::normalize(&config_bytes(&config)),
            Err(LagunaNormalizationError::UnsupportedQuantizationValue { .. })
        ));
    }

    for compressed_encoding in ["nvfp4", "packed-affine"] {
        let mut config = config_value(1);
        config["quantization_config"] = json!({
            "quant_method": compressed_encoding,
            "format": compressed_encoding
        });
        assert!(matches!(
            LagunaTargetNormalizer::normalize(&config_bytes(&config)),
            Err(LagunaNormalizationError::UnsupportedStorageEncoding { .. })
        ));
    }

    let mut incomplete_compressed_config = config_value(1);
    incomplete_compressed_config["quantization_config"] = json!({
        "quant_method": "compressed-tensors",
        "format": "nvfp4-pack-quantized"
    });
    assert!(matches!(
        LagunaTargetNormalizer::normalize(&config_bytes(&incomplete_compressed_config)),
        Err(LagunaNormalizationError::UnsupportedQuantizationValue { .. })
    ));
}
