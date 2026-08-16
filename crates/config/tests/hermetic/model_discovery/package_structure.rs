use std::fs;
use std::path::{Path, PathBuf};

#[test]
fn should_keep_neutral_discovery_orchestration_free_of_family_artifact_rules() {
    let discovery_source_directory = discovery_source_directory();
    let neutral_discovery_source = fs::read_to_string(discovery_source_directory.join("mod.rs"))
        .expect("neutral discovery source should be readable");
    let qwen_discovery_source = fs::read_to_string(discovery_source_directory.join("qwen3_5.rs"))
        .expect("Qwen discovery source should be readable");

    for qwen_artifact_rule in [
        "vision_tower.",
        "max_position_embeddings",
        "model.safetensors.index.json",
        "tokenizer.json",
        "contains_mtp_component",
    ] {
        assert!(
            !neutral_discovery_source.contains(qwen_artifact_rule),
            "neutral discovery must not own Qwen artifact rule {qwen_artifact_rule}"
        );
        assert!(
            qwen_discovery_source.contains(qwen_artifact_rule),
            "Qwen discovery must own artifact rule {qwen_artifact_rule}"
        );
    }
}

#[test]
fn should_keep_every_family_discovery_source_recursively_isolated() {
    let discovery_source_directory = discovery_source_directory();
    let mut violations = Vec::new();
    for source_file in rust_source_files_recursively(&discovery_source_directory) {
        let relative_source_file = source_file
            .strip_prefix(&discovery_source_directory)
            .expect("discovery source should remain under its owner");
        let source = fs::read_to_string(&source_file).expect("discovery source should be readable");
        let owned_family = family_owned_by_path(relative_source_file);
        for (family_name, family_markers) in [
            ("Qwen", ["qwen3_5", "Qwen3_5"]),
            ("Laguna", ["laguna", "Laguna"]),
            ("DeepSeek", ["deepseek_v4", "DeepSeekV4"]),
        ] {
            if owned_family == Some(family_name) {
                continue;
            }
            if family_markers
                .iter()
                .any(|family_marker| source.contains(family_marker))
                && !is_exact_family_dispatch_file(relative_source_file)
            {
                violations.push((relative_source_file.to_path_buf(), family_name));
            }
        }
    }
    assert!(
        violations.is_empty(),
        "discovery sources must not cross model-family ownership: {violations:#?}"
    );
}

fn discovery_source_directory() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("model_discovery")
}

fn family_owned_by_path(relative_source_file: &Path) -> Option<&'static str> {
    let first_component = relative_source_file
        .components()
        .next()
        .and_then(|component| component.as_os_str().to_str())
        .unwrap_or_default();
    if first_component.starts_with("qwen3_5") {
        Some("Qwen")
    } else if first_component.starts_with("laguna") {
        Some("Laguna")
    } else if first_component.starts_with("deepseek_v4") {
        Some("DeepSeek")
    } else {
        None
    }
}

fn is_exact_family_dispatch_file(relative_source_file: &Path) -> bool {
    relative_source_file == Path::new("mod.rs")
        || relative_source_file == Path::new("model_family.rs")
}

fn rust_source_files_recursively(source_directory: &Path) -> Vec<PathBuf> {
    let mut pending_directories = vec![source_directory.to_path_buf()];
    let mut source_files = Vec::new();
    while let Some(pending_directory) = pending_directories.pop() {
        let mut directory_entries = fs::read_dir(&pending_directory)
            .expect("discovery source directory should be readable")
            .map(|directory_entry| {
                directory_entry
                    .expect("discovery source entry should be readable")
                    .path()
            })
            .collect::<Vec<_>>();
        directory_entries.sort();
        for entry_path in directory_entries {
            if entry_path.is_dir() {
                pending_directories.push(entry_path);
            } else if entry_path
                .extension()
                .and_then(|extension| extension.to_str())
                == Some("rs")
            {
                source_files.push(entry_path);
            }
        }
    }
    source_files.sort();
    source_files
}
