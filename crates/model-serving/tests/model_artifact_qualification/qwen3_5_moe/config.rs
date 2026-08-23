use astronomical_model_serving::{OptiQMetadata, Qwen3_5Config};

#[test]
#[ignore = "requires model_directories to discover the Ornith 1.5 qualification artifact"]
fn should_parse_the_complete_local_ornith_config() {
    let model_directory = crate::common::configured_ornith_model_artifact_directory();
    let config_bytes = std::fs::read(model_directory.join("config.json"))
        .expect("the config.json should be readable for qualification");

    let metadata_bytes = std::fs::read(model_directory.join("optiq_metadata.json"))
        .expect("the optiq_metadata.json should be readable for qualification");

    let ornith_config = Qwen3_5Config::from_json_bytes(&config_bytes)
        .expect("the complete Ornith config should pass exact field validation");
    let optiq_metadata = OptiQMetadata::from_json_bytes(&metadata_bytes)
        .expect("the complete OptiQ metadata should pass exact field validation");
    optiq_metadata
        .validate_against_config(&ornith_config)
        .expect("the OptiQ metadata bit map should match the config");
}
