use astronomical_model_serving::{
    Qwen3_5FeedForwardArchitecture, Qwen3_5MtpPairingCompatibilityError,
    Qwen3_5StandaloneMtpArtifactValidator, Qwen3_5StandaloneMtpConfig,
    Qwen3_5StandaloneMtpConfigError, StandaloneMtpNamespaceError, TensorDtype, TensorProfile,
    compare_qwen3_5_mtp_pairing_contracts, normalize_qwen3_5_standalone_mtp_tensor_name,
    qwen3_5_mtp_tensor_profiles,
};
use std::collections::BTreeSet;
use std::io::Write;

use crate::common::qwen3_5::certified_dense_qwen3_6_config;

#[test]
fn should_normalize_every_published_standalone_mtp_root() {
    for stored_name in [
        "fc.weight",
        "layers.0.self_attn.q_proj.weight",
        "norm.weight",
        "pre_fc_norm_embedding.weight",
        "pre_fc_norm_hidden.weight",
    ] {
        assert_eq!(
            normalize_qwen3_5_standalone_mtp_tensor_name(stored_name)
                .expect("published standalone root should normalize"),
            format!("language_model.mtp.{stored_name}")
        );
    }
}

#[test]
fn should_reject_nonstandalone_or_nonterminal_layer_names() {
    for stored_name in [
        "language_model.mtp.fc.weight",
        "mtp.fc.weight",
        "layers.1.self_attn.q_proj.weight",
        "embed_tokens.weight",
        "lm_head.weight",
        "layers..0.weight",
    ] {
        assert!(matches!(
            normalize_qwen3_5_standalone_mtp_tensor_name(stored_name),
            Err(StandaloneMtpNamespaceError::UnsupportedNamespace { .. })
        ));
    }
}

#[test]
fn should_parse_native_and_affine_standalone_config_contracts() {
    let native_config = Qwen3_5StandaloneMtpConfig::from_json_bytes(
        standalone_config_json(None).to_string().as_bytes(),
    )
    .expect("native standalone config should parse");
    assert_eq!(native_config.maximum_draft_depth(), 2);
    assert_eq!(native_config.hidden_size(), 64);
    assert_eq!(native_config.vocabulary_size(), 128);
    assert_eq!(native_config.mtp_layer_count(), 1);
    assert_eq!(
        native_config.feed_forward_architecture(),
        Qwen3_5FeedForwardArchitecture::Dense
    );
    assert_eq!(native_config.quantization_profile(), None);

    let affine_document = serde_json::json!({ "bits": 4, "group_size": 32 });
    let affine_config = Qwen3_5StandaloneMtpConfig::from_json_bytes(
        standalone_config_json(Some(affine_document))
            .to_string()
            .as_bytes(),
    )
    .expect("affine standalone config should parse");
    let affine_profile = affine_config
        .quantization_profile()
        .expect("affine profile should be retained");
    assert_eq!(affine_profile.bits, 4);
    assert_eq!(affine_profile.group_size, 32);
}

#[test]
fn should_accept_root_tied_embedding_evidence_when_the_nested_copy_is_absent() {
    let mut config_document = standalone_config_json(None);
    config_document["tie_word_embeddings"] = serde_json::json!(true);
    config_document["text_config"]
        .as_object_mut()
        .expect("text config should be an object")
        .remove("tie_word_embeddings");

    let config =
        Qwen3_5StandaloneMtpConfig::from_json_bytes(config_document.to_string().as_bytes())
            .expect("root tied-embedding evidence should be sufficient");

    assert!(config.has_tied_embeddings());
}

#[test]
fn should_reject_conflicting_or_unsupported_standalone_config_evidence() {
    let mut conflicting_config = standalone_config_json(None);
    conflicting_config["quantization"] = serde_json::json!({ "bits": 4, "group_size": 32 });
    conflicting_config["quantization_config"] = serde_json::json!({ "bits": 5, "group_size": 32 });
    assert!(matches!(
        Qwen3_5StandaloneMtpConfig::from_json_bytes(conflicting_config.to_string().as_bytes()),
        Err(Qwen3_5StandaloneMtpConfigError::QuantizationDocumentDisagreement)
    ));

    let mut dedicated_embedding_config = standalone_config_json(None);
    dedicated_embedding_config["text_config"]["mtp_use_dedicated_embeddings"] =
        serde_json::json!(true);
    assert!(matches!(
        Qwen3_5StandaloneMtpConfig::from_json_bytes(
            dedicated_embedding_config.to_string().as_bytes()
        ),
        Err(Qwen3_5StandaloneMtpConfigError::DedicatedEmbeddingsUnsupported)
    ));
}

#[test]
fn should_validate_complete_native_standalone_storage_without_reading_payload_bytes() {
    let temporary_directory = tempfile::tempdir().expect("artifact root should be created");
    let model_directory = temporary_directory.path().join("Standalone-MTP");
    std::fs::create_dir_all(&model_directory).expect("artifact directory should be created");
    let mut target_config = certified_dense_qwen3_6_config();
    let native_weight_names = qwen3_5_mtp_tensor_profiles(&target_config)
        .into_iter()
        .filter(|profile| profile.name.ends_with(".weight"))
        .map(|profile| profile.name)
        .collect::<BTreeSet<_>>();
    target_config.resolve_unquantized_modules_from_shard_index(&native_weight_names);
    let native_profiles = qwen3_5_mtp_tensor_profiles(&target_config);
    let config_json = standalone_config_for_target(&target_config);
    std::fs::write(model_directory.join("config.json"), config_json.to_string())
        .expect("standalone config should be written");
    std::fs::write(model_directory.join("tokenizer.json"), "{}")
        .expect("standalone tokenizer should be written");
    write_sparse_safetensors(&model_directory.join("model.safetensors"), &native_profiles);

    let validated_artifact = Qwen3_5StandaloneMtpArtifactValidator::new(
        &target_config,
        "Standalone-MTP",
        "revision-one",
    )
    .validate(&model_directory)
    .expect("complete native standalone artifact should validate");

    assert_eq!(validated_artifact.model_id(), "Standalone-MTP");
    assert_eq!(validated_artifact.source_count(), 1);
    assert_eq!(validated_artifact.tensor_profiles().len(), 15);
    assert!(validated_artifact.total_payload_bytes() > 0);
    assert_eq!(validated_artifact.storage_fingerprint().len(), 64);
    let changed_revision_artifact = Qwen3_5StandaloneMtpArtifactValidator::new(
        &target_config,
        "Standalone-MTP",
        "revision-two",
    )
    .validate(&model_directory)
    .expect("unchanged storage should validate under changed provenance");
    assert_ne!(
        validated_artifact.storage_fingerprint(),
        changed_revision_artifact.storage_fingerprint()
    );
}

#[test]
fn should_validate_complete_affine_indexed_standalone_storage() {
    let temporary_directory = tempfile::tempdir().expect("artifact root should be created");
    let model_directory = temporary_directory.path().join("Affine-MTP");
    std::fs::create_dir_all(&model_directory).expect("artifact directory should be created");
    let target_config = certified_dense_qwen3_6_config();
    let affine_profiles = qwen3_5_mtp_tensor_profiles(&target_config);
    let mut config_json = standalone_config_for_target(&target_config);
    let quantization_document = serde_json::json!({
        "bits": target_config.default_quantization_bits(),
        "group_size": target_config.default_quantization_group_size()
    });
    config_json["quantization"] = quantization_document.clone();
    config_json["quantization_config"] = quantization_document;
    std::fs::write(model_directory.join("config.json"), config_json.to_string())
        .expect("standalone config should be written");
    std::fs::write(model_directory.join("tokenizer.json"), "{}")
        .expect("standalone tokenizer should be written");
    let shard_file_name = "model-00001.safetensors";
    write_sparse_safetensors(&model_directory.join(shard_file_name), &affine_profiles);
    let weight_map = affine_profiles
        .iter()
        .map(|profile| {
            (
                profile
                    .name
                    .strip_prefix("language_model.mtp.")
                    .expect("MTP profile should use canonical prefix"),
                shard_file_name,
            )
        })
        .collect::<std::collections::BTreeMap<_, _>>();
    std::fs::write(
        model_directory.join("model.safetensors.index.json"),
        serde_json::json!({ "weight_map": weight_map }).to_string(),
    )
    .expect("standalone index should be written");

    let validated_artifact =
        Qwen3_5StandaloneMtpArtifactValidator::new(&target_config, "Affine-MTP", "affine-revision")
            .validate(&model_directory)
            .expect("complete affine indexed artifact should validate");

    assert_eq!(validated_artifact.source_count(), 1);
    assert_eq!(validated_artifact.tensor_profiles().len(), 31);
    assert!(
        validated_artifact
            .tensor_profiles()
            .iter()
            .any(|profile| profile.dtype == TensorDtype::UInt32)
    );
}

#[test]
fn should_accept_a_single_file_with_an_exact_index_for_that_same_file() {
    let temporary_directory = tempfile::tempdir().expect("artifact root should be created");
    let mut target_config = certified_dense_qwen3_6_config();
    let model_directory = temporary_directory.path();
    let native_weight_names = qwen3_5_mtp_tensor_profiles(&target_config)
        .into_iter()
        .filter(|profile| profile.name.ends_with(".weight"))
        .map(|profile| profile.name)
        .collect::<BTreeSet<_>>();
    target_config.resolve_unquantized_modules_from_shard_index(&native_weight_names);
    let profiles = qwen3_5_mtp_tensor_profiles(&target_config);
    std::fs::write(
        model_directory.join("config.json"),
        standalone_config_for_target(&target_config).to_string(),
    )
    .expect("standalone config should be written");
    std::fs::write(model_directory.join("tokenizer.json"), "{}")
        .expect("standalone tokenizer should be written");
    write_sparse_safetensors(&model_directory.join("model.safetensors"), &profiles);
    let weight_map = profiles
        .iter()
        .map(|profile| {
            (
                profile
                    .name
                    .strip_prefix("language_model.mtp.")
                    .expect("MTP profile should use canonical prefix"),
                "model.safetensors",
            )
        })
        .collect::<std::collections::BTreeMap<_, _>>();
    std::fs::write(
        model_directory.join("model.safetensors.index.json"),
        serde_json::json!({ "weight_map": weight_map }).to_string(),
    )
    .expect("standalone index should be written");

    Qwen3_5StandaloneMtpArtifactValidator::new(&target_config, "indexed-single", "revision")
        .validate(model_directory)
        .expect("an exact index over model.safetensors should validate");
}

#[test]
fn should_prove_complete_target_drafter_geometry_tokenizer_and_depth_compatibility() {
    let target_config = certified_dense_qwen3_6_config();
    let drafter_config = Qwen3_5StandaloneMtpConfig::from_json_bytes(
        standalone_config_for_target(&target_config)
            .to_string()
            .as_bytes(),
    )
    .expect("compatible standalone config should parse");
    let tokenizer_bytes = tiny_tokenizer_bytes(&[("<unk>", 0), ("Romeo", 1), ("Juliet", 2)]);

    let compatibility = compare_qwen3_5_mtp_pairing_contracts(
        &target_config,
        &tokenizer_bytes,
        &drafter_config,
        &tokenizer_bytes,
        Some(2),
    )
    .expect("complete matching contracts should be compatible");

    assert_eq!(compatibility.maximum_draft_depth, 2);
    assert_eq!(compatibility.requested_draft_depth, 2);

    let incompatible_tokenizer = tiny_tokenizer_bytes(&[("<unk>", 0), ("Romeo", 2), ("Juliet", 1)]);
    assert_eq!(
        compare_qwen3_5_mtp_pairing_contracts(
            &target_config,
            &tokenizer_bytes,
            &drafter_config,
            &incompatible_tokenizer,
            Some(2),
        ),
        Err(Qwen3_5MtpPairingCompatibilityError::TokenizerMappingMismatch)
    );
    assert_eq!(
        compare_qwen3_5_mtp_pairing_contracts(
            &target_config,
            &tokenizer_bytes,
            &drafter_config,
            &tokenizer_bytes,
            Some(3),
        ),
        Err(Qwen3_5MtpPairingCompatibilityError::UnsupportedDraftDepth)
    );
}

fn standalone_config_json(quantization_document: Option<serde_json::Value>) -> serde_json::Value {
    let mut config = serde_json::json!({
        "model_type": "qwen3_5_mtp",
        "block_size": 3,
        "tie_word_embeddings": false,
        "text_config": {
            "model_type": "qwen3_5_text",
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "vocab_size": 128,
            "mtp_num_hidden_layers": 1,
            "mtp_use_dedicated_embeddings": false,
            "tie_word_embeddings": false
        }
    });
    if let Some(quantization_document) = quantization_document {
        config["quantization"] = quantization_document.clone();
        config["quantization_config"] = quantization_document;
    }
    config
}

fn standalone_config_for_target(
    target_config: &astronomical_model_serving::Qwen3_5Config,
) -> serde_json::Value {
    serde_json::json!({
        "model_type": "qwen3_5_mtp",
        "block_size": 3,
        "tie_word_embeddings": target_config.has_tied_embeddings(),
        "text_config": {
            "model_type": "qwen3_5_text",
            "hidden_size": target_config.hidden_size(),
            "intermediate_size": target_config.dense_intermediate_size(),
            "num_attention_heads": target_config.query_head_count(),
            "num_key_value_heads": target_config.key_value_head_count(),
            "head_dim": target_config.head_dimension(),
            "vocab_size": target_config.vocabulary_size(),
            "mtp_num_hidden_layers": 1,
            "mtp_use_dedicated_embeddings": false,
            "tie_word_embeddings": target_config.has_tied_embeddings(),
            "num_hidden_layers": target_config.layer_count(),
            "max_position_embeddings": target_config.maximum_position_count(),
            "attention_bias": target_config.has_attention_bias(),
            "hidden_act": target_config.hidden_activation(),
            "rms_norm_eps": f32::from_bits(target_config.rms_norm_epsilon_bits()),
            "partial_rotary_factor": f32::from_bits(target_config.partial_rotary_factor_bits()),
            "linear_conv_kernel_dim": target_config.linear_convolution_kernel_dimension(),
            "linear_num_key_heads": target_config.linear_key_head_count(),
            "linear_num_value_heads": target_config.linear_value_head_count(),
            "linear_key_head_dim": target_config.linear_key_head_dimension(),
            "linear_value_head_dim": target_config.linear_value_head_dimension(),
            "layer_types": target_config.layer_types(),
            "rope_parameters": {
                "rope_theta": f32::from_bits(target_config.rope_theta_bits())
            }
        }
    })
}

fn tiny_tokenizer_bytes(vocabulary: &[(&str, u32)]) -> Vec<u8> {
    let vocabulary = vocabulary
        .iter()
        .map(|(token, identifier)| ((*token).to_owned(), serde_json::json!(identifier)))
        .collect::<serde_json::Map<_, _>>();
    serde_json::to_vec(&serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": {"type": "WhitespaceSplit"},
        "post_processor": null,
        "decoder": null,
        "model": {
            "type": "WordLevel",
            "vocab": vocabulary,
            "unk_token": "<unk>"
        }
    }))
    .expect("tokenizer fixture should serialize")
}

fn write_sparse_safetensors(file_path: &std::path::Path, profiles: &[TensorProfile]) {
    let mut header_entries = serde_json::Map::new();
    let mut payload_offset_bytes = 0_u64;
    for profile in profiles {
        let stored_name = profile
            .name
            .strip_prefix("language_model.mtp.")
            .expect("MTP profile should use the canonical prefix");
        let element_bytes = match profile.dtype {
            TensorDtype::ModelFloat
            | TensorDtype::AffineQuantizationFloat
            | TensorDtype::BFloat16 => 2_u64,
            TensorDtype::Float32 | TensorDtype::UInt32 => 4_u64,
        };
        let tensor_element_count = profile.shape.iter().fold(1_u64, |count, dimension| {
            count
                .checked_mul(*dimension as u64)
                .expect("test tensor element count should fit")
        });
        let tensor_payload_bytes = tensor_element_count
            .checked_mul(element_bytes)
            .expect("test tensor payload should fit");
        let next_payload_offset = payload_offset_bytes
            .checked_add(tensor_payload_bytes)
            .expect("test payload offset should fit");
        header_entries.insert(
            stored_name.to_owned(),
            serde_json::json!({
                "dtype": match profile.dtype {
                    TensorDtype::UInt32 => "U32",
                    TensorDtype::Float32 => "F32",
                    _ => "BF16",
                },
                "shape": profile.shape,
                "data_offsets": [payload_offset_bytes, next_payload_offset]
            }),
        );
        payload_offset_bytes = next_payload_offset;
    }
    let header_bytes = serde_json::to_vec(&header_entries).expect("header should serialize");
    let mut safetensors_file =
        std::fs::File::create(file_path).expect("safetensors fixture should be created");
    safetensors_file
        .write_all(&(header_bytes.len() as u64).to_le_bytes())
        .expect("header length should be written");
    safetensors_file
        .write_all(&header_bytes)
        .expect("header should be written");
    safetensors_file
        .set_len(8 + header_bytes.len() as u64 + payload_offset_bytes)
        .expect("sparse payload length should be established");
}
