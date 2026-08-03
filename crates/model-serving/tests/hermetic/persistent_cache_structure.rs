use std::path::PathBuf;

#[test]
fn should_keep_all_ssd_backed_cache_implementation_under_the_root_persistent_cache_package() {
    let model_serving_source_directory = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");
    let root_persistent_cache_directory = model_serving_source_directory.join("persistent_cache");
    let qwen_source_directory = model_serving_source_directory.join("qwen3_5_moe");

    assert!(
        root_persistent_cache_directory.is_dir(),
        "SSD-backed cache implementation must live under {}",
        root_persistent_cache_directory.display()
    );
    assert!(
        !model_serving_source_directory
            .join("persistent_decoder_cache")
            .exists(),
        "model-serving must not retain the retired persistent_decoder_cache package"
    );
    for retired_qwen_persistence_directory_name in [
        "persistent_prompt_cache",
        "persistent_visual_embedding_cache",
    ] {
        assert!(
            !qwen_source_directory
                .join(retired_qwen_persistence_directory_name)
                .exists(),
            "Qwen must not retain the SSD persistence directory {retired_qwen_persistence_directory_name}"
        );
    }
}
