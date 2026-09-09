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
