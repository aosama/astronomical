use astronomical_model_serving::{OptiQMetadata, Qwen3_5MoEConfig};

#[test]
#[ignore = "requires model_directories to discover Ornith-1.0-35B-OptiQ-4bit"]
fn should_parse_the_complete_pinned_local_ornith_config() {
    let model_directory = crate::common::configured_ornith_model_artifact_directory();
    let config_bytes = std::fs::read(model_directory.join("config.json"))
        .expect("the pinned config.json should be readable for qualification");

    let metadata_bytes = std::fs::read(model_directory.join("optiq_metadata.json"))
        .expect("the pinned optiq_metadata.json should be readable for qualification");

    let ornith_config = Qwen3_5MoEConfig::from_json_bytes(&config_bytes)
        .expect("the complete pinned Ornith config should pass exact field validation");
    let optiq_metadata = OptiQMetadata::from_json_bytes(&metadata_bytes)
        .expect("the complete pinned OptiQ metadata should pass exact field validation");
    optiq_metadata
        .validate_against_config(&ornith_config)
        .expect("the pinned OptiQ metadata bit map should exactly match the config");
}
