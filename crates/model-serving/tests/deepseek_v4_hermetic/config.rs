use astronomical_model_serving::{DeepSeekV4DsparkArtifactCapability, DeepSeekV4Flash0731Config};

use super::support::selected_config_value;

#[test]
fn should_accept_the_selected_target_only_configuration() {
    let config_bytes = serde_json::to_vec(&selected_config_value(false))
        .expect("the target-only configuration should serialize");

    let deepseek_config = DeepSeekV4Flash0731Config::from_json_bytes(&config_bytes)
        .expect("the selected target-only configuration should validate");

    assert_eq!(
        deepseek_config.dspark_artifact_capability(),
        &DeepSeekV4DsparkArtifactCapability::TargetOnly
    );
}

#[test]
fn should_accept_the_selected_dspark_configuration() {
    let config_bytes = serde_json::to_vec(&selected_config_value(true))
        .expect("the DSpark configuration should serialize");

    let deepseek_config = DeepSeekV4Flash0731Config::from_json_bytes(&config_bytes)
        .expect("the selected DSpark configuration should validate");

    assert!(deepseek_config.dspark_artifact_capability().is_declared());
}

#[test]
fn should_reject_another_model_family_or_incomplete_dspark_metadata() {
    let mut wrong_family_config = selected_config_value(false);
    wrong_family_config["model_type"] = serde_json::json!("qwen3_5_moe");
    let wrong_family_config_bytes = serde_json::to_vec(&wrong_family_config)
        .expect("the wrong-family configuration should serialize");
    assert!(DeepSeekV4Flash0731Config::from_json_bytes(&wrong_family_config_bytes).is_err());

    let mut incomplete_dspark_config = selected_config_value(true);
    incomplete_dspark_config
        .as_object_mut()
        .expect("the configuration should be an object")
        .remove("dspark_markov_rank");
    let incomplete_dspark_config_bytes = serde_json::to_vec(&incomplete_dspark_config)
        .expect("the incomplete DSpark configuration should serialize");
    assert!(DeepSeekV4Flash0731Config::from_json_bytes(&incomplete_dspark_config_bytes).is_err());
}
