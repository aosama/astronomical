use super::{recommended_sampling_for, write_recommended_generation_config};

fn read_generation_config(artifact_directory: &std::path::Path) -> serde_json::Value {
    let config_text = std::fs::read_to_string(artifact_directory.join("generation_config.json"))
        .expect("the generation config should be readable");
    serde_json::from_str(&config_text).expect("the generation config should parse")
}

#[test]
fn should_write_the_qwen_recipe_when_the_source_published_none() {
    let directory = tempfile::tempdir().expect("a temporary directory should be creatable");

    let was_written = write_recommended_generation_config(directory.path(), "Qwen/Qwen3.5-9B")
        .expect("the recipe should be written");

    let generation_config = read_generation_config(directory.path());
    assert!(was_written);
    assert_eq!(generation_config["temperature"], 0.7);
    assert_eq!(generation_config["top_k"], 20);
    assert_eq!(generation_config["presence_penalty"], 1.5);
}

#[test]
fn should_never_override_a_value_the_source_published() {
    let directory = tempfile::tempdir().expect("a temporary directory should be creatable");
    std::fs::write(
        directory.path().join("generation_config.json"),
        r#"{"temperature": 0.2, "eos_token_id": 7}"#,
    )
    .expect("the published config should be writable");

    write_recommended_generation_config(directory.path(), "qwen3.6-27b")
        .expect("the merge should be written");

    let generation_config = read_generation_config(directory.path());
    assert_eq!(generation_config["temperature"], 0.2);
    assert_eq!(generation_config["eos_token_id"], 7);
    assert_eq!(generation_config["top_p"], 0.8);
}

#[test]
fn should_write_nothing_for_a_family_without_a_recipe() {
    let directory = tempfile::tempdir().expect("a temporary directory should be creatable");

    let was_written = write_recommended_generation_config(directory.path(), "google/gemma-4-e2b")
        .expect("an unknown family is not an error");

    assert!(!was_written);
    assert!(!directory.path().join("generation_config.json").exists());
}

#[test]
fn should_read_the_family_from_the_last_path_segment_only() {
    assert!(recommended_sampling_for("Qwen/Qwen3.5-0.8B").is_some());
    assert!(recommended_sampling_for("openbmb/MiniCPM5-1B").is_some());
    assert!(recommended_sampling_for("qwen3.5-lab/another-model").is_none());
    assert!(recommended_sampling_for("").is_none());
}

#[test]
fn should_not_rewrite_a_file_that_already_holds_the_whole_recipe() {
    let directory = tempfile::tempdir().expect("a temporary directory should be creatable");
    write_recommended_generation_config(directory.path(), "Qwen3.5-4B")
        .expect("the first write should succeed");

    let was_rewritten = write_recommended_generation_config(directory.path(), "Qwen3.5-4B")
        .expect("the second call should succeed");

    assert!(!was_rewritten);
}
