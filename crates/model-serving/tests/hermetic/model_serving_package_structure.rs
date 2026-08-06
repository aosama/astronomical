use std::path::{Path, PathBuf};

#[test]
fn should_group_architecture_neutral_model_serving_modules_by_domain_ownership() {
    let model_serving_source_directory = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");

    assert_source_directory_contains_files(
        &model_serving_source_directory,
        "artifact_validation",
        &[
            "mod.rs",
            "bounded_safetensors.rs",
            "error.rs",
            "required_files.rs",
            "safetensors_dtype.rs",
            "types.rs",
            "validated_artifact.rs",
        ],
    );
    assert_source_directory_contains_files(
        &model_serving_source_directory,
        "safetensors",
        &["mod.rs", "header.rs"],
    );
    assert_source_directory_contains_files(
        &model_serving_source_directory,
        "memory",
        &[
            "mod.rs",
            "adaptive_ram_growth_guard.rs",
            "mlx_memory_telemetry.rs",
        ],
    );
    assert_source_directory_contains_files(
        &model_serving_source_directory,
        "engine_backed_worker",
        &[
            "mod.rs",
            "construction.rs",
            "fatal.rs",
            "output.rs",
            "protocol.rs",
            "support.rs",
            "generation_advance.rs",
            "generation_start.rs",
            "idle_command.rs",
            "memory_limit.rs",
            "model_swap.rs",
        ],
    );
    assert_source_directory_contains_files(
        &model_serving_source_directory,
        "expert_paging",
        &[
            "mod.rs",
            "bounded_expert_reader.rs",
            "expert_cache.rs",
            "expert_cache_capacity.rs",
            "expert_cache_eviction.rs",
            "expert_cache_pressure.rs",
            "expert_cache_statistics.rs",
            "memory_budget.rs",
            "quantized_expert_manifest.rs",
            "quantized_expert_validation.rs",
            "safetensors_header.rs",
            "source_manifests.rs",
        ],
    );
    assert_source_directory_contains_files(
        &model_serving_source_directory,
        "model_family_runtime",
        &[
            "mod.rs",
            "inference_engine.rs",
            "processor.rs",
            "request.rs",
            "output.rs",
        ],
    );

    for retired_flat_source_file_name in [
        "adaptive_ram_growth_guard.rs",
        "bounded_safetensors.rs",
        "bounded_safetensors_header.rs",
        "engine_backed_worker_construction.rs",
        "engine_backed_worker_fatal.rs",
        "engine_backed_worker_output.rs",
        "engine_backed_worker_protocol.rs",
        "engine_backed_worker_support.rs",
        "error.rs",
        "mlx_memory_telemetry.rs",
        "required_files.rs",
        "safetensors_dtype.rs",
        "types.rs",
        "validated_artifact.rs",
    ] {
        assert!(
            !model_serving_source_directory
                .join(retired_flat_source_file_name)
                .exists(),
            "retired flat model-serving source file must not remain: {retired_flat_source_file_name}"
        );
    }

    for intentional_root_source_file_name in [
        "lib.rs",
        "model_generation_processor.rs",
        "performance_attribution.rs",
    ] {
        assert!(
            model_serving_source_directory
                .join(intentional_root_source_file_name)
                .is_file(),
            "intentional model-serving root source file must remain: {intentional_root_source_file_name}"
        );
    }

    let shared_expert_paging_source_directory =
        model_serving_source_directory.join("expert_paging");
    for shared_expert_paging_source_file_name in [
        "mod.rs",
        "bounded_expert_reader.rs",
        "expert_cache.rs",
        "expert_cache_capacity.rs",
        "expert_cache_eviction.rs",
        "expert_cache_pressure.rs",
        "expert_cache_statistics.rs",
        "memory_budget.rs",
        "quantized_expert_manifest.rs",
        "quantized_expert_validation.rs",
        "safetensors_header.rs",
        "source_manifests.rs",
    ] {
        let shared_expert_paging_source = std::fs::read_to_string(
            shared_expert_paging_source_directory.join(shared_expert_paging_source_file_name),
        )
        .expect("shared expert-paging source must be readable");
        assert!(
            !shared_expert_paging_source.contains("qwen3_5")
                && !shared_expert_paging_source.contains("deepseek_v4"),
            "shared expert-paging source must not depend on a concrete model family: {shared_expert_paging_source_file_name}"
        );
    }
}

fn assert_source_directory_contains_files(
    model_serving_source_directory: &Path,
    source_directory_name: &str,
    required_source_file_names: &[&str],
) {
    let source_directory = model_serving_source_directory.join(source_directory_name);
    assert!(
        source_directory.is_dir(),
        "model-serving domain directory must exist: {source_directory_name}"
    );
    for required_source_file_name in required_source_file_names {
        assert!(
            source_directory.join(required_source_file_name).is_file(),
            "model-serving domain file must exist: {source_directory_name}/{required_source_file_name}"
        );
    }
}
