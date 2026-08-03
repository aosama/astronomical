use std::path::PathBuf;

#[test]
fn should_group_qwen_modules_by_their_concrete_domain_concern() {
    let qwen_source_directory = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("qwen3_5_moe");

    for required_qwen_package_name in [
        "artifacts",
        "configuration",
        "decoder",
        "inference_execution",
        "model",
        "text",
        "vision",
    ] {
        assert!(
            qwen_source_directory
                .join(required_qwen_package_name)
                .is_dir(),
            "Qwen package {required_qwen_package_name} must exist"
        );
    }

    for retired_flat_qwen_module_name in [
        "artifact.rs",
        "config.rs",
        "decoder_cache_layout.rs",
        "engine_request.rs",
        "engine_decoder_state_reuse.rs",
        "engine_visual_embeddings.rs",
        "image_processor.rs",
        "model.rs",
        "prefill_chunck_sizer.rs",
        "prompt.rs",
        "tokenizer.rs",
        "vision_model.rs",
    ] {
        assert!(
            !qwen_source_directory
                .join(retired_flat_qwen_module_name)
                .exists(),
            "Qwen module {retired_flat_qwen_module_name} must live in its domain package"
        );
    }

    let inference_engine_source_directory = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("inference_engine");
    assert!(
        inference_engine_source_directory.is_dir(),
        "model-serving must own architecture-neutral inference execution separately from Qwen"
    );
    for architecture_neutral_source_file_name in ["contract.rs", "mlx_owner.rs"] {
        let architecture_neutral_source = std::fs::read_to_string(
            inference_engine_source_directory.join(architecture_neutral_source_file_name),
        )
        .expect("architecture-neutral inference source must be readable");
        assert!(
            !architecture_neutral_source.contains("qwen3_5"),
            "architecture-neutral inference source must not depend on Qwen"
        );
    }
    assert!(
        !qwen_source_directory.join("engine").join("mod.rs").exists(),
        "Qwen must not own an engine module after inference execution is separated"
    );
}
