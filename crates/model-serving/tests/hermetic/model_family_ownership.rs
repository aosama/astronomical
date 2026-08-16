use std::path::PathBuf;

use super::source_package_guard::{
    ConcreteModelFamily, families_mentioned_by_source, rust_source_files_recursively,
};

#[test]
fn should_keep_existing_concrete_model_family_roots_isolated() {
    let source_directory = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");
    for (family_roots, owned_family) in [
        (
            vec![
                source_directory.join("qwen3_5"),
                source_directory.join("qwen3_5_moe"),
            ],
            ConcreteModelFamily::Qwen,
        ),
        (
            vec![source_directory.join("deepseek_v4")],
            ConcreteModelFamily::DeepSeekV4,
        ),
    ] {
        for family_root in family_roots {
            assert!(family_root.is_dir(), "family source root must exist");
            for source_file in rust_source_files_recursively(&family_root) {
                let foreign_families = families_mentioned_by_source(&source_file)
                    .into_iter()
                    .filter(|mentioned_family| *mentioned_family != owned_family)
                    .collect::<Vec<_>>();
                assert!(
                    foreign_families.is_empty(),
                    "family source must not import another family: {source_file:?}: {foreign_families:?}"
                );
            }
        }
    }
}

#[test]
fn should_keep_neutral_model_serving_sources_free_of_concrete_families() {
    let source_directory = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");
    let concrete_family_roots = ["qwen3_5", "qwen3_5_moe", "laguna", "deepseek_v4"];
    let dispatch_files = [
        PathBuf::from("lib.rs"),
        PathBuf::from("model_family_runtime/inference_engine.rs"),
        PathBuf::from("model_family_runtime/output.rs"),
        PathBuf::from("model_family_runtime/processor.rs"),
        PathBuf::from("model_family_runtime/request.rs"),
    ];
    let mut violations = Vec::new();

    for source_file in rust_source_files_recursively(&source_directory) {
        let relative_source_file = source_file
            .strip_prefix(&source_directory)
            .expect("source must remain beneath model-serving src");
        let root_name = relative_source_file
            .components()
            .next()
            .and_then(|component| component.as_os_str().to_str())
            .unwrap_or_default();
        if concrete_family_roots.contains(&root_name)
            || dispatch_files.contains(&relative_source_file.to_path_buf())
        {
            continue;
        }
        let mentioned_families = families_mentioned_by_source(&source_file);
        if !mentioned_families.is_empty() {
            violations.push((relative_source_file.to_path_buf(), mentioned_families));
        }
    }
    assert!(
        violations.is_empty(),
        "neutral sources must remain family-free: {violations:#?}"
    );
}
