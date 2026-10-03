use std::fs;

use astronomical_model_serving::{
    K2HorizonMoVAArtifactValidator, K2HorizonMoVAWeightDialect,
    expected_stacked_affine_tensor_names,
};

use super::support::{family_member_config_json, write_stacked_affine_fixture};

#[test]
fn should_validate_a_tiny_stacked_affine_family_member() {
    let temporary_directory = tempfile::tempdir().expect("temp dir");
    let model_directory = write_stacked_affine_fixture(temporary_directory.path());
    let validated = K2HorizonMoVAArtifactValidator::new()
        .validate(&model_directory)
        .expect("tiny stacked affine fixture should validate");
    assert!(validated.shard_count() > 0);
    assert_eq!(
        validated.total_payload_bytes(),
        fs::metadata(model_directory.join("model-00001-of-00001.safetensors"))
            .expect("shard metadata")
            .len()
    );
    assert_eq!(validated.config().num_hidden_layers(), 2);
    assert!(!expected_stacked_affine_tensor_names(validated.config()).is_empty());
}

#[test]
fn should_prefer_only_an_immutable_provenance_revision_over_the_config_hash() {
    let temporary_directory = tempfile::tempdir().expect("temp dir");
    let model_directory = write_stacked_affine_fixture(temporary_directory.path());
    let without_provenance = K2HorizonMoVAArtifactValidator::new()
        .validate(&model_directory)
        .expect("the fixture should validate without a provenance file");
    let config_hash_revision = without_provenance.revision().to_owned();
    fs::write(
        model_directory.join(".astronomical-library-provenance.json"),
        r#"{"provider_model_id":"example/model","revision":"0c576733b69e","schema_version":1}"#,
    )
    .expect("truncated provenance file should be written");
    let with_truncated_revision = K2HorizonMoVAArtifactValidator::new()
        .validate(&model_directory)
        .expect("the fixture should validate with a truncated provenance file");
    assert_eq!(
        config_hash_revision,
        with_truncated_revision.revision(),
        "a truncated provenance revision is not an immutable published \
         identity and must fall back to the config hash"
    );

    let immutable_revision = "0c576733b69e17e4f9c1b0d2a3c4d5e6f708192a";
    fs::write(
        model_directory.join(".astronomical-library-provenance.json"),
        format!(
            r#"{{"provider_model_id":"example/model","revision":"{immutable_revision}","schema_version":1}}"#
        ),
    )
    .expect("immutable provenance file should be written");
    let with_provenance = K2HorizonMoVAArtifactValidator::new()
        .validate(&model_directory)
        .expect("the fixture should validate with a provenance file");
    assert_eq!(
        with_provenance.revision(),
        immutable_revision,
        "a 40-character lowercase hexadecimal provenance revision must take \
         precedence over the config hash"
    );
}

#[test]
fn should_reject_unstacked_per_expert_tensors() {
    let temporary_directory = tempfile::tempdir().expect("temp dir");
    let model_directory = write_stacked_affine_fixture(temporary_directory.path());
    let unstacked_index = serde_json::json!({
        "weight_map": {
            "model.layers.1.mlp.experts.0.up_proj.weight": "model-00001-of-00001.safetensors",
            "model.layers.1.self_attn.v_experts.0.weight": "model-00001-of-00001.safetensors"
        }
    });
    fs::write(
        model_directory.join("model.safetensors.index.json"),
        unstacked_index.to_string(),
    )
    .expect("unstacked index should be written");
    assert_eq!(
        K2HorizonMoVAWeightDialect::from_tensor_names([
            "model.layers.1.mlp.experts.0.up_proj.weight",
            "model.layers.1.self_attn.v_experts.0.weight",
        ]),
        K2HorizonMoVAWeightDialect::UnstackedPerExpert
    );
    assert!(
        K2HorizonMoVAArtifactValidator::new()
            .validate(&model_directory)
            .is_err()
    );
}

#[test]
fn should_reject_a_missing_indexed_shard() {
    let temporary_directory = tempfile::tempdir().expect("temp dir");
    let model_directory = write_stacked_affine_fixture(temporary_directory.path());
    fs::remove_file(model_directory.join("model-00001-of-00001.safetensors"))
        .expect("shard should be removed");
    assert!(
        K2HorizonMoVAArtifactValidator::new()
            .validate(&model_directory)
            .is_err()
    );
}

#[test]
fn should_reject_wrong_model_type_directories() {
    let temporary_directory = tempfile::tempdir().expect("temp dir");
    let model_directory = write_stacked_affine_fixture(temporary_directory.path());
    fs::write(
        model_directory.join("config.json"),
        family_member_config_json(2, &[0], 4, 2).replace("k2_horizon_mova", "llama"),
    )
    .expect("wrong type config should be written");
    assert!(
        K2HorizonMoVAArtifactValidator::new()
            .validate(&model_directory)
            .is_err()
    );
}

#[test]
fn should_reject_sparse_feed_forward_members_as_not_executable_yet() {
    let temporary_directory = tempfile::tempdir().expect("temp dir");
    let model_directory = write_stacked_affine_fixture(temporary_directory.path());
    fs::write(
        model_directory.join("config.json"),
        family_member_config_json(2, &[0], 4, 0),
    )
    .expect("sparse feed-forward config should be written");
    let validation_error = K2HorizonMoVAArtifactValidator::new()
        .validate(&model_directory)
        .expect_err("a sparse feed-forward member has no executable serving path yet");
    assert!(
        validation_error.to_string().contains("not executable yet"),
        "the rejection must say the member is not executable: {validation_error}"
    );
}
