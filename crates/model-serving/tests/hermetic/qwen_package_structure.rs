use std::path::PathBuf;

#[test]
fn should_group_qwen_modules_by_their_concrete_domain_concern() {
    let model_serving_source_directory = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");
    let shared_qwen_source_directory = model_serving_source_directory.join("qwen3_5");
    let sparse_qwen_source_directory = model_serving_source_directory.join("qwen3_5_moe");

    for required_shared_qwen_package_name in [
        "artifacts",
        "configuration",
        "decoder",
        "dense",
        "inference_execution",
        "model",
        "quantizations",
        "text",
        "vision",
    ] {
        assert!(
            shared_qwen_source_directory
                .join(required_shared_qwen_package_name)
                .is_dir(),
            "Shared Qwen package {required_shared_qwen_package_name} must exist"
        );
    }

    for required_sparse_qwen_package_name in ["artifacts", "expert_paging", "model"] {
        assert!(
            sparse_qwen_source_directory
                .join(required_sparse_qwen_package_name)
                .is_dir(),
            "Sparse Qwen package {required_sparse_qwen_package_name} must exist"
        );
    }

    let qwen_expert_paging_source_directory = sparse_qwen_source_directory.join("expert_paging");
    for required_qwen_expert_paging_package_name in ["expert_pager"] {
        assert!(
            qwen_expert_paging_source_directory
                .join(required_qwen_expert_paging_package_name)
                .is_dir(),
            "Qwen expert-paging package {required_qwen_expert_paging_package_name} must exist"
        );
    }
    for required_qwen_expert_paging_source_file_name in [
        "mod.rs",
        "expert_pager_construction.rs",
        "quantized_expert_layer_plan.rs",
    ] {
        assert!(
            qwen_expert_paging_source_directory
                .join(required_qwen_expert_paging_source_file_name)
                .is_file(),
            "Qwen expert-paging source must remain family-owned: {required_qwen_expert_paging_source_file_name}"
        );
    }
    let qwen_expert_pager_source_directory =
        qwen_expert_paging_source_directory.join("expert_pager");
    for required_qwen_expert_pager_source_file_name in ["mod.rs", "rust_layer_streaming.rs"] {
        assert!(
            qwen_expert_pager_source_directory
                .join(required_qwen_expert_pager_source_file_name)
                .is_file(),
            "Qwen expert-pager source must exist: {required_qwen_expert_pager_source_file_name}"
        );
    }
    assert!(
        !qwen_expert_paging_source_directory
            .join("expert_cache_page_assembly.rs")
            .exists(),
        "direct page-table execution must not retain selected-page assembly"
    );

    for shared_only_qwen_package_name in [
        "configuration",
        "decoder",
        "dense",
        "inference_execution",
        "quantizations",
        "text",
        "vision",
    ] {
        assert!(
            !sparse_qwen_source_directory
                .join(shared_only_qwen_package_name)
                .exists(),
            "Sparse Qwen package must not own {shared_only_qwen_package_name}"
        );
    }
    assert!(
        !shared_qwen_source_directory.join("expert_paging").exists(),
        "Shared Qwen package must not own sparse expert paging"
    );

    for shared_expert_paging_source_name in [
        "expert_cache_statistics.rs",
        "memory_budget.rs",
        "quantized_expert_manifest.rs",
        "quantized_expert_validation.rs",
        "safetensors_header.rs",
        "source_manifests.rs",
    ] {
        assert!(
            !qwen_expert_paging_source_directory
                .join(shared_expert_paging_source_name)
                .exists(),
            "shared expert-paging source must not remain under Qwen ownership: {shared_expert_paging_source_name}"
        );
    }

    let model_memory_admission_source = std::fs::read_to_string(
        shared_qwen_source_directory
            .join("model")
            .join("memory_admission.rs"),
    )
    .expect("Qwen model memory-admission source must be readable");
    assert!(
        !model_memory_admission_source.contains("impl super::super::inference_execution"),
        "Qwen model memory admission must not own inference-execution implementation blocks"
    );
    assert!(
        shared_qwen_source_directory
            .join("inference_execution")
            .join("memory_admission.rs")
            .is_file(),
        "Qwen inference execution must own adaptive memory-admission orchestration"
    );
    for separated_memory_execution_owner in [
        "completed_forward_memory.rs",
        "prefill_capacity_recovery.rs",
    ] {
        assert!(
            shared_qwen_source_directory
                .join("inference_execution")
                .join(separated_memory_execution_owner)
                .is_file(),
            "Qwen inference execution must keep memory responsibilities separated: {separated_memory_execution_owner}"
        );
    }
    assert!(
        shared_qwen_source_directory
            .join("inference_execution")
            .join("prefill_chunck_sizer")
            .join("persisted_state.rs")
            .is_file(),
        "Qwen prefill chunk sizing must isolate persisted-state construction"
    );

    let sparse_model_module_source =
        std::fs::read_to_string(sparse_qwen_source_directory.join("model").join("mod.rs"))
            .expect("sparse Qwen model module source must be readable");
    assert!(
        !sparse_model_module_source.contains("pub(crate) use crate::qwen3_5"),
        "sparse Qwen modules must import explicit shared contracts without re-export bridges"
    );
    let shared_model_module_source =
        std::fs::read_to_string(shared_qwen_source_directory.join("model").join("mod.rs"))
            .expect("shared Qwen model module source must be readable");
    assert!(
        !shared_model_module_source.contains("pub(crate) use crate::qwen3_5_moe"),
        "shared Qwen modules must import explicit sparse contracts without re-export bridges"
    );

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
            !shared_qwen_source_directory
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
        !shared_qwen_source_directory
            .join("engine")
            .join("mod.rs")
            .exists(),
        "Qwen must not own an engine module after inference execution is separated"
    );
}
