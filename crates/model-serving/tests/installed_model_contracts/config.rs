use astronomical_model_serving::{OptiQMetadata, Qwen3_5Config};

#[test]
#[ignore = "requires model_directories to discover the large sparse MoE installed model"]
fn should_parse_the_large_sparse_moe_config() {
    let model_directory = crate::common::configured_large_sparse_moe_model_directory();
    let config_bytes = std::fs::read(model_directory.join("config.json"))
        .expect("the config.json should be readable for acceptance");

    let ornith_config = Qwen3_5Config::from_json_bytes(&config_bytes)
        .expect("the complete Ornith config should pass exact field validation");
    assert!(ornith_config.layer_count() > 0);
    assert!(ornith_config.expert_count() > 0);
    assert!(ornith_config.experts_per_token() > 0);
    assert!(
        [2, 3, 4, 5, 6, 8].contains(&ornith_config.default_quantization_bits()),
        "default affine bit width must be a supported MLX width"
    );
    assert!(
        [32u32, 64, 128].contains(&ornith_config.default_quantization_group_size()),
        "default affine group size must be a supported MLX group size"
    );

    // Some OptiQ packages ship root metadata; mixed static packages do not.
    // Production serving reads config.json either way.
    let metadata_path = model_directory.join("optiq_metadata.json");
    if metadata_path.is_file() {
        let metadata_bytes = std::fs::read(&metadata_path)
            .expect("readable OptiQ metadata should load when the file is present");
        let optiq_metadata = OptiQMetadata::from_json_bytes(&metadata_bytes)
            .expect("present OptiQ metadata should pass exact field validation");
        optiq_metadata
            .validate_against_config(&ornith_config)
            .expect("present OptiQ metadata bit map should match the config");
    }
}
