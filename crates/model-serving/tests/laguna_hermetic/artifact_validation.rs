use std::fs::{self, File};
use std::os::unix::fs::FileExt;

use astronomical_model_serving::{
    ArtifactValidationError, LagunaArtifactValidationError, LagunaArtifactValidator,
    LagunaAttentionProjection, LagunaCanonicalTensorAssemblyKind, LagunaExpertProjection,
    LagunaGlobalTensorRole, LagunaLayerTensorRole, LagunaShardIndexError, LagunaTensorComponent,
    LagunaTensorId, LagunaTensorNameNormalizationError,
};
use safetensors::Dtype;

use super::artifact_support::{
    FIRST_SHARD_FILE_NAME, SECOND_SHARD_FILE_NAME, SyntheticLagunaArtifact, SyntheticTensor,
};

#[test]
fn should_validate_a_complete_dense_directory_into_one_canonical_tensor_contract() {
    let model_directory = tempfile::tempdir().expect("the test should create a model directory");
    let fixture = SyntheticLagunaArtifact::dense("");
    fixture.write(model_directory.path());

    let validated_artifact = LagunaArtifactValidator::new()
        .validate(model_directory.path())
        .expect("the complete dense Laguna artifact should validate");

    assert_eq!(
        validated_artifact.target_contract().model().layer_count(),
        1
    );
    assert_eq!(validated_artifact.total_tensor_payload_bytes(), 800);
    assert_eq!(
        validated_artifact.total_shard_file_bytes(),
        fixture.serialized_shard_file_bytes()
    );
    assert_eq!(validated_artifact.tensor_contract().descriptors().len(), 14);
    let query_descriptor = validated_artifact
        .tensor_contract()
        .descriptor(&layer_id(LagunaLayerTensorRole::Attention(
            LagunaAttentionProjection::Query,
        )))
        .expect("the canonical query descriptor should exist");
    assert_eq!(query_descriptor.logical_shape(), &[4, 4]);
    assert_eq!(query_descriptor.execution_dtype(), Dtype::F32);
    assert_eq!(query_descriptor.storage_dtype(), Dtype::F32);
    assert_eq!(
        query_descriptor.assembly_kind(),
        LagunaCanonicalTensorAssemblyKind::DirectAlias
    );
}

#[test]
fn should_accept_bare_and_wrapped_namespaces_without_leaking_names_into_tensor_ids() {
    for namespace_prefix in ["", "language_model."] {
        let model_directory =
            tempfile::tempdir().expect("the test should create a model directory");
        SyntheticLagunaArtifact::dense(namespace_prefix).write(model_directory.path());

        let validated_artifact = LagunaArtifactValidator::new()
            .validate(model_directory.path())
            .expect("both evidenced Laguna namespaces should validate");
        assert!(
            validated_artifact
                .tensor_contract()
                .descriptor(&LagunaTensorId::Global {
                    role: LagunaGlobalTensorRole::TokenEmbedding,
                    component: LagunaTensorComponent::Weight,
                })
                .is_some()
        );
        assert!(
            validated_artifact
                .tensor_contract()
                .descriptors()
                .keys()
                .all(|tensor_id| { !format!("{tensor_id:?}").contains("language_model") })
        );
    }
}

#[test]
fn should_produce_a_deterministic_metadata_only_storage_fingerprint() {
    let model_directory = tempfile::tempdir().expect("the test should create a model directory");
    SyntheticLagunaArtifact::dense("").write(model_directory.path());

    let first_fingerprint = *LagunaArtifactValidator::new()
        .validate(model_directory.path())
        .expect("the first validation should succeed")
        .storage_fingerprint();
    let second_fingerprint = *LagunaArtifactValidator::new()
        .validate(model_directory.path())
        .expect("the second validation should succeed")
        .storage_fingerprint();

    assert_eq!(first_fingerprint, second_fingerprint);
    assert_ne!(first_fingerprint, [0_u8; 32]);
}

#[test]
fn should_retain_exact_intervals_and_transfer_open_descriptors_after_paths_disappear() {
    let model_directory = tempfile::tempdir().expect("the test should create a model directory");
    SyntheticLagunaArtifact::dense("").write(model_directory.path());
    let validated_artifact = LagunaArtifactValidator::new()
        .validate(model_directory.path())
        .expect("the complete dense artifact should validate");
    let embedding_descriptor = validated_artifact
        .tensor_contract()
        .descriptor(&LagunaTensorId::Global {
            role: LagunaGlobalTensorRole::TokenEmbedding,
            component: LagunaTensorComponent::Weight,
        })
        .expect("the embedding descriptor should exist");
    let embedding_source = embedding_descriptor
        .sources()
        .first()
        .expect("a direct tensor should retain one source");
    assert_eq!(
        embedding_source.data_end_offset_bytes() - embedding_source.data_start_offset_bytes(),
        embedding_source.payload_bytes()
    );
    assert_eq!(embedding_source.raw_shape(), &[8, 4]);
    let expected_start_offset_bytes = embedding_source.data_start_offset_bytes();
    let expected_end_offset_bytes = embedding_source.data_end_offset_bytes();
    let expected_raw_tensor_name = embedding_source.raw_tensor_name().to_owned();

    let retained_files = validated_artifact
        .into_retained_files()
        .expect("validated descriptors should transfer");
    fs::remove_dir_all(model_directory.path()).expect("the fixture paths should be removable");
    assert!(!read_retained_file(retained_files.config_file()).is_empty());
    assert!(!read_retained_file(retained_files.index_file()).is_empty());
    let retained_inventory = retained_files
        .shard_files()
        .get(FIRST_SHARD_FILE_NAME)
        .expect("the first retained shard should transfer")
        .read_raw_safetensors_inventory_for_tests()
        .expect("the retained shard should remain readable");
    assert_eq!(retained_inventory.tensor_descriptors.len(), 7);
    let retained_embedding = retained_inventory
        .tensor_descriptors
        .iter()
        .find(|tensor| tensor.tensor_name == expected_raw_tensor_name)
        .expect("the retained inventory should contain the embedding source");
    assert_eq!(
        retained_embedding.data_start_offset_bytes,
        expected_start_offset_bytes
    );
    assert_eq!(
        retained_embedding.data_end_offset_bytes,
        expected_end_offset_bytes
    );
}

#[test]
fn should_reject_duplicate_index_keys_with_a_typed_cause() {
    let model_directory = tempfile::tempdir().expect("the test should create a model directory");
    let fixture = SyntheticLagunaArtifact::dense("");
    let serialized_shard_file_bytes = fixture.serialized_shard_file_bytes();
    fixture.write(model_directory.path());
    let duplicate_name = "model.embed_tokens.weight";
    let duplicate_index = format!(
        "{{\"metadata\":{{\"total_size\":{serialized_shard_file_bytes}}},\"weight_map\":{{\"{duplicate_name}\":\"{FIRST_SHARD_FILE_NAME}\",\"{duplicate_name}\":\"{FIRST_SHARD_FILE_NAME}\"}}}}"
    );
    fs::write(
        model_directory.path().join("model.safetensors.index.json"),
        duplicate_index,
    )
    .expect("the duplicate index should be written");

    assert!(matches!(
        validation_error(model_directory.path()),
        LagunaArtifactValidationError::ShardIndex(
            LagunaShardIndexError::DuplicateTensorName { .. }
        )
    ));
}

#[test]
fn should_reject_unsafe_or_missing_shard_files_with_typed_causes() {
    let unsafe_directory = tempfile::tempdir().expect("the test should create a model directory");
    let mut unsafe_fixture = SyntheticLagunaArtifact::dense("");
    unsafe_fixture.indexed_shard_by_tensor.insert(
        "model.embed_tokens.weight".to_owned(),
        "../outside.safetensors".to_owned(),
    );
    unsafe_fixture.write(unsafe_directory.path());
    assert!(matches!(
        validation_error(unsafe_directory.path()),
        LagunaArtifactValidationError::ShardIndex(
            LagunaShardIndexError::UnsafeShardFileName { .. }
        )
    ));

    let missing_directory = tempfile::tempdir().expect("the test should create a model directory");
    let missing_fixture = SyntheticLagunaArtifact::dense("");
    missing_fixture.write(missing_directory.path());
    fs::remove_file(missing_directory.path().join(FIRST_SHARD_FILE_NAME))
        .expect("the indexed shard should be removed");
    assert!(matches!(
        validation_error(missing_directory.path()),
        LagunaArtifactValidationError::MissingShard { .. }
    ));
}

#[test]
fn should_reject_index_and_physical_ownership_disagreement() {
    let missing_tensor_directory =
        tempfile::tempdir().expect("the test should create a model directory");
    let mut missing_tensor_fixture = SyntheticLagunaArtifact::dense("");
    missing_tensor_fixture.remove_physical_tensor("model.layers.0.self_attn.q_proj.weight");
    missing_tensor_fixture.write(missing_tensor_directory.path());
    assert!(matches!(
        validation_error(missing_tensor_directory.path()),
        LagunaArtifactValidationError::IndexedTensorMissing { .. }
    ));

    let wrong_shard_directory =
        tempfile::tempdir().expect("the test should create a model directory");
    let mut wrong_shard_fixture = SyntheticLagunaArtifact::dense("");
    wrong_shard_fixture.indexed_shard_by_tensor.insert(
        "model.embed_tokens.weight".to_owned(),
        SECOND_SHARD_FILE_NAME.to_owned(),
    );
    wrong_shard_fixture.write(wrong_shard_directory.path());
    assert!(matches!(
        validation_error(wrong_shard_directory.path()),
        LagunaArtifactValidationError::PhysicalTensorInWrongShard { .. }
    ));

    let unindexed_directory =
        tempfile::tempdir().expect("the test should create a model directory");
    let mut unindexed_fixture = SyntheticLagunaArtifact::dense("");
    unindexed_fixture
        .tensors_by_shard
        .get_mut(FIRST_SHARD_FILE_NAME)
        .expect("the first shard should exist")
        .push(SyntheticTensor {
            name: "model.layers.0.self_attn.extra.weight".to_owned(),
            dtype: "F32",
            shape: vec![1],
        });
    unindexed_fixture.write(unindexed_directory.path());
    assert!(matches!(
        validation_error(unindexed_directory.path()),
        LagunaArtifactValidationError::PhysicalTensorNotIndexed { .. }
    ));
}

#[test]
fn should_reject_a_duplicate_physical_tensor_across_shards() {
    let model_directory = tempfile::tempdir().expect("the test should create a model directory");
    let mut fixture = SyntheticLagunaArtifact::dense("");
    let duplicate_tensor = fixture
        .tensors_by_shard
        .get(FIRST_SHARD_FILE_NAME)
        .expect("the first shard should exist")
        .first()
        .expect("the first shard should contain a tensor")
        .clone();
    fixture
        .tensors_by_shard
        .get_mut(SECOND_SHARD_FILE_NAME)
        .expect("the second shard should exist")
        .push(duplicate_tensor);
    fixture.write(model_directory.path());

    assert!(matches!(
        validation_error(model_directory.path()),
        LagunaArtifactValidationError::DuplicatePhysicalTensor { .. }
    ));
}

#[test]
fn should_reject_unknown_executable_tensors() {
    let unknown_directory = tempfile::tempdir().expect("the test should create a model directory");
    let mut unknown_fixture = SyntheticLagunaArtifact::dense("");
    let unknown_tensor = unknown_fixture
        .tensors_by_shard
        .get(FIRST_SHARD_FILE_NAME)
        .expect("the first shard should exist")
        .first()
        .expect("the first shard should contain a tensor")
        .clone();
    let mut unknown_tensor = unknown_tensor;
    unknown_tensor.name = "model.layers.0.future.weight".to_owned();
    unknown_fixture
        .tensors_by_shard
        .get_mut(FIRST_SHARD_FILE_NAME)
        .expect("the first shard should exist")
        .push(unknown_tensor.clone());
    unknown_fixture
        .indexed_shard_by_tensor
        .insert(unknown_tensor.name, FIRST_SHARD_FILE_NAME.to_owned());
    unknown_fixture.write(unknown_directory.path());
    assert!(matches!(
        validation_error(unknown_directory.path()),
        LagunaArtifactValidationError::TensorNames(
            LagunaTensorNameNormalizationError::UnknownTensorName { .. }
        )
    ));
}

#[test]
fn should_reject_missing_expected_tensors_and_wrong_shape_or_dtype() {
    let missing_directory = tempfile::tempdir().expect("the test should create a model directory");
    let mut missing_fixture = SyntheticLagunaArtifact::dense("");
    missing_fixture.remove_tensor_completely("model.layers.0.self_attn.q_norm.weight");
    missing_fixture.write(missing_directory.path());
    assert!(matches!(
        validation_error(missing_directory.path()),
        LagunaArtifactValidationError::ExpectedTensorMissing { .. }
    ));

    let shape_directory = tempfile::tempdir().expect("the test should create a model directory");
    let mut shape_fixture = SyntheticLagunaArtifact::dense("");
    shape_fixture
        .tensor_mut("model.layers.0.self_attn.q_proj.weight")
        .shape = vec![3, 4];
    shape_fixture.write(shape_directory.path());
    assert!(matches!(
        validation_error(shape_directory.path()),
        LagunaArtifactValidationError::TensorShapeMismatch { .. }
    ));

    let dtype_directory = tempfile::tempdir().expect("the test should create a model directory");
    let mut dtype_fixture = SyntheticLagunaArtifact::dense("");
    dtype_fixture
        .tensor_mut("model.layers.0.self_attn.q_proj.weight")
        .dtype = "I32";
    dtype_fixture.write(dtype_directory.path());
    assert!(matches!(
        validation_error(dtype_directory.path()),
        LagunaArtifactValidationError::TensorDtypeMismatch { .. }
    ));
}

#[test]
fn should_separate_execution_dtype_from_supported_physical_float_storage() {
    let model_directory = tempfile::tempdir().expect("the test should create a model directory");
    let mut fixture = SyntheticLagunaArtifact::dense("");
    fixture.config["torch_dtype"] = serde_json::json!("float32");
    fixture
        .tensor_mut("model.layers.0.self_attn.q_proj.weight")
        .dtype = "BF16";
    fixture.write(model_directory.path());

    let artifact = LagunaArtifactValidator::new()
        .validate(model_directory.path())
        .expect("supported physical float storage should remain independent from execution");
    let query_weight = artifact
        .tensor_contract()
        .descriptor(&layer_id(LagunaLayerTensorRole::Attention(
            LagunaAttentionProjection::Query,
        )))
        .expect("the query weight should exist");
    assert_eq!(query_weight.execution_dtype(), Dtype::F32);
    assert_eq!(query_weight.storage_dtype(), Dtype::BF16);
    assert_eq!(query_weight.sources()[0].raw_dtype(), Dtype::BF16);
}

#[test]
fn should_preserve_neutral_interval_validation_as_a_typed_cause() {
    let model_directory = tempfile::tempdir().expect("the test should create a model directory");
    SyntheticLagunaArtifact::dense("").write(model_directory.path());
    let malformed_header =
        r#"{"model.embed_tokens.weight":{"dtype":"F32","shape":[8,4],"data_offsets":[1,129]}}"#;
    let mut malformed_shard = Vec::new();
    malformed_shard.extend_from_slice(
        &u64::try_from(malformed_header.len())
            .expect("the malformed fixture header length should fit u64")
            .to_le_bytes(),
    );
    malformed_shard.extend_from_slice(malformed_header.as_bytes());
    malformed_shard.extend_from_slice(&vec![0_u8; 129]);
    fs::write(
        model_directory.path().join(FIRST_SHARD_FILE_NAME),
        malformed_shard,
    )
    .expect("the malformed shard should be written");

    assert!(matches!(
        validation_error(model_directory.path()),
        LagunaArtifactValidationError::Artifact(
            ArtifactValidationError::SafetensorsInvalidDataOffsets { .. }
        )
    ));
}

#[test]
fn should_validate_a_stacked_sparse_layer_with_router_and_correction_bias() {
    let model_directory = tempfile::tempdir().expect("the test should create a model directory");
    SyntheticLagunaArtifact::sparse_stacked().write(model_directory.path());

    let validated_artifact = LagunaArtifactValidator::new()
        .validate(model_directory.path())
        .expect("the evidenced stacked sparse assembly should validate");
    let routed_gate = validated_artifact
        .tensor_contract()
        .descriptor(&layer_id(LagunaLayerTensorRole::RoutedExpert(
            LagunaExpertProjection::Gate,
        )))
        .expect("the routed gate descriptor should exist");
    assert_eq!(routed_gate.logical_shape(), &[2, 3, 4]);
    assert_eq!(
        routed_gate.assembly_kind(),
        LagunaCanonicalTensorAssemblyKind::StackedSource
    );
    assert!(
        validated_artifact
            .tensor_contract()
            .descriptor(&layer_id(LagunaLayerTensorRole::RouterCorrectionBias,))
            .is_some()
    );
}

#[test]
fn should_validate_tied_output_head_absence_and_reject_its_presence() {
    let valid_directory = tempfile::tempdir().expect("the test should create a model directory");
    let mut valid_fixture = SyntheticLagunaArtifact::dense("");
    valid_fixture.config["tie_word_embeddings"] = serde_json::json!(true);
    valid_fixture.remove_tensor_completely("lm_head.weight");
    valid_fixture.write(valid_directory.path());
    let valid_artifact = LagunaArtifactValidator::new()
        .validate(valid_directory.path())
        .expect("a tied artifact should omit the output head");
    assert!(
        valid_artifact
            .tensor_contract()
            .descriptor(&LagunaTensorId::Global {
                role: LagunaGlobalTensorRole::OutputHead,
                component: LagunaTensorComponent::Weight,
            })
            .is_none()
    );

    let invalid_directory = tempfile::tempdir().expect("the test should create a model directory");
    let mut invalid_fixture = SyntheticLagunaArtifact::dense("");
    invalid_fixture.config["tie_word_embeddings"] = serde_json::json!(true);
    invalid_fixture.write(invalid_directory.path());
    assert!(matches!(
        validation_error(invalid_directory.path()),
        LagunaArtifactValidationError::UnexpectedCanonicalTensor { .. }
    ));
}

#[test]
fn should_validate_per_head_and_per_element_attention_gate_shapes() {
    for (gating_type, gate_shape) in [("per_head", vec![2, 4]), ("per_element", vec![4, 4])] {
        let model_directory =
            tempfile::tempdir().expect("the test should create a model directory");
        let mut fixture = SyntheticLagunaArtifact::dense("");
        fixture.config["gating_types"] = serde_json::json!([gating_type]);
        let gate_tensor_name = "model.layers.0.self_attn.g_proj.weight";
        fixture
            .tensors_by_shard
            .get_mut(FIRST_SHARD_FILE_NAME)
            .expect("the first shard should exist")
            .push(SyntheticTensor {
                name: gate_tensor_name.to_owned(),
                dtype: "F32",
                shape: gate_shape.clone(),
            });
        fixture.indexed_shard_by_tensor.insert(
            gate_tensor_name.to_owned(),
            FIRST_SHARD_FILE_NAME.to_owned(),
        );
        fixture.write(model_directory.path());

        let artifact = LagunaArtifactValidator::new()
            .validate(model_directory.path())
            .expect("the configured attention gate shape should validate");
        assert_eq!(
            artifact
                .tensor_contract()
                .descriptor(&layer_id(LagunaLayerTensorRole::Attention(
                    LagunaAttentionProjection::Gate,
                )))
                .expect("the attention gate descriptor should exist")
                .logical_shape(),
            gate_shape
        );
    }
}

#[test]
fn should_resolve_legacy_boolean_gating_as_per_head_only_with_matching_inventory() {
    let valid_directory = tempfile::tempdir().expect("the test should create a model directory");
    let mut valid_fixture = SyntheticLagunaArtifact::dense("");
    valid_fixture.config["gating"] = serde_json::json!(true);
    add_attention_gate(&mut valid_fixture, vec![2, 4]);
    valid_fixture.write(valid_directory.path());
    let valid_artifact = LagunaArtifactValidator::new()
        .validate(valid_directory.path())
        .expect("legacy boolean gating and per-head storage should resolve together");
    assert_eq!(
        valid_artifact.target_contract().layers()[0]
            .attention()
            .gating_kind(),
        astronomical_model_serving::LagunaGatingKind::PerHead
    );

    let invalid_directory = tempfile::tempdir().expect("the test should create a model directory");
    let mut invalid_fixture = SyntheticLagunaArtifact::dense("");
    invalid_fixture.config["gating"] = serde_json::json!(true);
    add_attention_gate(&mut invalid_fixture, vec![4, 4]);
    invalid_fixture.write(invalid_directory.path());
    assert!(matches!(
        validation_error(invalid_directory.path()),
        LagunaArtifactValidationError::TensorShapeMismatch { .. }
    ));
}

fn validation_error(model_directory: &std::path::Path) -> LagunaArtifactValidationError {
    LagunaArtifactValidator::new()
        .validate(model_directory)
        .expect_err("the malformed synthetic Laguna artifact should fail")
}

fn layer_id(role: LagunaLayerTensorRole) -> LagunaTensorId {
    LagunaTensorId::Layer {
        layer_index: 0,
        role,
        component: LagunaTensorComponent::Weight,
    }
}

fn add_attention_gate(fixture: &mut SyntheticLagunaArtifact, gate_shape: Vec<usize>) {
    let gate_tensor_name = "model.layers.0.self_attn.g_proj.weight";
    fixture
        .tensors_by_shard
        .get_mut(FIRST_SHARD_FILE_NAME)
        .expect("the first shard should exist")
        .push(SyntheticTensor {
            name: gate_tensor_name.to_owned(),
            dtype: "F32",
            shape: gate_shape,
        });
    fixture.indexed_shard_by_tensor.insert(
        gate_tensor_name.to_owned(),
        FIRST_SHARD_FILE_NAME.to_owned(),
    );
}

fn read_retained_file(retained_file: &File) -> Vec<u8> {
    let file_size = usize::try_from(
        retained_file
            .metadata()
            .expect("the retained descriptor metadata should remain readable")
            .len(),
    )
    .expect("the synthetic retained file size should fit usize");
    let mut file_bytes = vec![0_u8; file_size];
    retained_file
        .read_exact_at(&mut file_bytes, 0)
        .expect("the retained descriptor bytes should remain readable");
    file_bytes
}
