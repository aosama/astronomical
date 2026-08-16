use std::fs;
use std::path::{Path, PathBuf};

#[test]
fn should_keep_neutral_worker_sources_free_of_concrete_family_ownership() {
    let worker_source_directory = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");
    let allowed_concrete_source_files = [
        "lib.rs",
        "model_family_factory.rs",
        "qwen3_5_model_startup.rs",
        "qwen3_5_model_startup_error.rs",
    ];
    let mut violations = Vec::new();
    for worker_source_file in rust_source_files_recursively(&worker_source_directory) {
        let relative_source_file = worker_source_file
            .strip_prefix(&worker_source_directory)
            .expect("worker source should remain under src");
        if relative_source_file
            .to_str()
            .is_some_and(|relative_path| allowed_concrete_source_files.contains(&relative_path))
        {
            continue;
        }
        let worker_source =
            fs::read_to_string(&worker_source_file).expect("worker source should be readable");
        for concrete_family_identifier in [
            "qwen3_5",
            "Qwen3_5",
            "laguna",
            "Laguna",
            "deepseek_v4",
            "DeepSeekV4",
        ] {
            if worker_source.contains(concrete_family_identifier) {
                violations.push((
                    relative_source_file.to_path_buf(),
                    concrete_family_identifier,
                ));
            }
        }
    }
    assert!(
        violations.is_empty(),
        "neutral worker sources must not own concrete-family behavior: {violations:#?}"
    );
}

#[test]
fn should_keep_qwen_startup_isolated_from_other_model_families() {
    let worker_source_directory = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");
    for qwen_source_file_name in ["qwen3_5_model_startup.rs", "qwen3_5_model_startup_error.rs"] {
        let qwen_source = fs::read_to_string(worker_source_directory.join(qwen_source_file_name))
            .expect("Qwen startup source should be readable");
        for foreign_family_identifier in ["laguna", "Laguna", "deepseek_v4", "DeepSeekV4"] {
            assert!(
                !qwen_source.contains(foreign_family_identifier),
                "Qwen startup must not own {foreign_family_identifier}: {qwen_source_file_name}"
            );
        }
    }
}

#[test]
fn should_keep_qwen_chunk_sizer_construction_in_qwen_startup() {
    let worker_source_directory = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");
    let worker_startup_source =
        fs::read_to_string(worker_source_directory.join("worker_startup.rs"))
            .expect("worker startup source should be readable");
    let family_factory_source =
        fs::read_to_string(worker_source_directory.join("model_family_factory.rs"))
            .expect("model family factory source should be readable");
    let qwen_startup_source =
        fs::read_to_string(worker_source_directory.join("qwen3_5_model_startup.rs"))
            .expect("Qwen startup source should be readable");

    for neutral_source in [&worker_startup_source, &family_factory_source] {
        assert!(!neutral_source.contains("Qwen3_5PromptProcessingChunkSizer"));
        assert!(!neutral_source.contains("prompt_processing_chunk_sizer_override"));
    }
    assert!(qwen_startup_source.contains("Qwen3_5PromptProcessingChunkSizer"));
}

#[test]
fn should_keep_laguna_classified_but_non_executable_at_the_staged_boundary() {
    let worker_source_directory = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");
    let family_factory_source =
        fs::read_to_string(worker_source_directory.join("model_family_factory.rs"))
            .expect("model family factory source should be readable");

    assert!(family_factory_source.contains("Some(ModelFamily::Laguna)"));
    assert!(family_factory_source.contains("Laguna model family is not executable yet"));
    assert!(!family_factory_source.contains("initialize_laguna_model"));
}

fn rust_source_files_recursively(source_directory: &Path) -> Vec<PathBuf> {
    let mut pending_directories = vec![source_directory.to_path_buf()];
    let mut source_files = Vec::new();
    while let Some(pending_directory) = pending_directories.pop() {
        let mut directory_entries = fs::read_dir(&pending_directory)
            .expect("worker source directory should be readable")
            .map(|directory_entry| {
                directory_entry
                    .expect("worker source entry should be readable")
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
