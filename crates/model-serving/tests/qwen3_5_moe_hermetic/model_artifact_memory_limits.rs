use std::{fs, path::PathBuf};

const MODEL_ARTIFACT_TEST_RELATIVE_DIRECTORY: &str = "tests/model_artifact_qualification";
const FORBIDDEN_MODEL_ARTIFACT_ACTIVE_LIMIT_LITERAL: &str = "30 * 1024 * 1024 * 1024";
const FORBIDDEN_MODEL_ARTIFACT_ALLOCATOR_CACHE_LIMIT_DECLARATION: &str =
    "ORNITH_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES: usize = 1024 * 1024 * 1024";

#[test]
fn should_use_the_configured_model_artifact_memory_ceiling_when_it_is_lower_than_the_machine() {
    assert_eq!(
        crate::common::resolve_model_artifact_qualification_mlx_memory_ceiling_bytes(
            Some(35_000_000_000),
            40_000_000_000,
        ),
        35_000_000_000,
    );
}

#[test]
fn should_never_raise_model_artifact_memory_above_the_machine_ceiling() {
    assert_eq!(
        crate::common::resolve_model_artifact_qualification_mlx_memory_ceiling_bytes(
            Some(45_000_000_000),
            40_000_000_000,
        ),
        40_000_000_000,
    );
}

#[test]
fn should_use_the_machine_model_artifact_memory_ceiling_when_no_user_limit_exists() {
    assert_eq!(
        crate::common::resolve_model_artifact_qualification_mlx_memory_ceiling_bytes(
            None,
            40_000_000_000,
        ),
        40_000_000_000,
    );
}

#[test]
fn should_keep_model_artifact_tests_from_using_literal_ornith_memory_budgets() {
    let model_artifact_test_directory =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(MODEL_ARTIFACT_TEST_RELATIVE_DIRECTORY);
    let model_artifact_test_sources =
        fs::read_dir(&model_artifact_test_directory).unwrap_or_else(|read_directory_error| {
            panic!(
                "should read {}: {read_directory_error}",
                model_artifact_test_directory.display()
            )
        });

    let mut offending_source_descriptions = Vec::new();
    for model_artifact_test_source_entry in model_artifact_test_sources {
        let model_artifact_test_source_path = model_artifact_test_source_entry
            .unwrap_or_else(|directory_entry_error| {
                panic!(
                    "should read an entry from {}: {directory_entry_error}",
                    model_artifact_test_directory.display()
                )
            })
            .path();
        if model_artifact_test_source_path
            .extension()
            .and_then(|extension| extension.to_str())
            != Some("rs")
        {
            continue;
        }

        let model_artifact_test_source_text = fs::read_to_string(&model_artifact_test_source_path)
            .unwrap_or_else(|read_source_error| {
                panic!(
                    "should read {}: {read_source_error}",
                    model_artifact_test_source_path.display()
                )
            });
        for (source_line_index, model_artifact_test_source_line) in
            model_artifact_test_source_text.lines().enumerate()
        {
            let source_line_number = source_line_index + 1;
            if model_artifact_test_source_line
                .contains(FORBIDDEN_MODEL_ARTIFACT_ACTIVE_LIMIT_LITERAL)
            {
                offending_source_descriptions.push(format!(
                    "{}:{} contains literal active-memory budget {}",
                    model_artifact_test_source_path.display(),
                    source_line_number,
                    FORBIDDEN_MODEL_ARTIFACT_ACTIVE_LIMIT_LITERAL
                ));
            }
            if model_artifact_test_source_line
                .contains(FORBIDDEN_MODEL_ARTIFACT_ALLOCATOR_CACHE_LIMIT_DECLARATION)
            {
                offending_source_descriptions.push(format!(
                    "{}:{} contains literal allocator-cache memory budget declaration {}",
                    model_artifact_test_source_path.display(),
                    source_line_number,
                    FORBIDDEN_MODEL_ARTIFACT_ALLOCATOR_CACHE_LIMIT_DECLARATION
                ));
            }
        }
    }

    assert!(
        offending_source_descriptions.is_empty(),
        "ignored model-artifact tests must derive MLX memory limits from the machine GPU wired-memory limit instead of fixed Ornith budgets:\n{}",
        offending_source_descriptions.join("\n")
    );
}
