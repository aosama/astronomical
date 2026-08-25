//! Hermetic file-boundary coverage for the optional bounded Qwen thinking seed.

use astronomical_ipc_protocol::MAX_QWEN_THINKING_CHANNEL_SEED_BYTES;
use astronomical_supervisor::load_qwen_thinking_channel_seed;

#[tokio::test]
async fn should_load_thinking_markdown_from_a_temp_instance_state_directory() {
    let instance_state_directory =
        tempfile::tempdir().expect("a temporary instance state directory should be created");
    let thinking_seed_file_path = instance_state_directory.path().join("thinking.md");
    std::fs::write(
        &thinking_seed_file_path,
        "Two households, both alike in dignity, in Romeo and Juliet.\n",
    )
    .expect("thinking markdown should be written");

    assert_eq!(
        load_qwen_thinking_channel_seed(true, &thinking_seed_file_path)
            .await
            .as_deref(),
        Some("Two households, both alike in dignity, in Romeo and Juliet.")
    );
}

#[tokio::test]
async fn should_treat_a_missing_thinking_markdown_file_as_absent() {
    let instance_state_directory =
        tempfile::tempdir().expect("a temporary instance state directory should be created");
    let missing_thinking_seed_file_path = instance_state_directory.path().join("thinking.md");

    assert_eq!(
        load_qwen_thinking_channel_seed(true, &missing_thinking_seed_file_path).await,
        None
    );
}

#[tokio::test]
async fn should_treat_whitespace_only_thinking_markdown_as_absent() {
    let instance_state_directory =
        tempfile::tempdir().expect("a temporary instance state directory should be created");
    let thinking_seed_file_path = instance_state_directory.path().join("thinking.md");
    std::fs::write(&thinking_seed_file_path, "  \n\t\n")
        .expect("whitespace-only thinking markdown should be written");

    assert_eq!(
        load_qwen_thinking_channel_seed(true, &thinking_seed_file_path).await,
        None
    );
}

#[tokio::test]
async fn should_ignore_existing_thinking_markdown_when_the_experiment_is_disabled() {
    let instance_state_directory =
        tempfile::tempdir().expect("a temporary instance state directory should be created");
    let thinking_seed_file_path = instance_state_directory.path().join("thinking.md");
    std::fs::write(
        &thinking_seed_file_path,
        "Two households, both alike in dignity, in Romeo and Juliet.\n",
    )
    .expect("thinking markdown should be written");

    assert_eq!(
        load_qwen_thinking_channel_seed(false, &thinking_seed_file_path).await,
        None
    );
}

#[tokio::test]
async fn should_ignore_thinking_markdown_that_exceeds_the_worker_boundary() {
    let instance_state_directory =
        tempfile::tempdir().expect("a temporary instance state directory should be created");
    let thinking_seed_file_path = instance_state_directory.path().join("thinking.md");
    std::fs::write(
        &thinking_seed_file_path,
        "R".repeat(MAX_QWEN_THINKING_CHANNEL_SEED_BYTES + 1),
    )
    .expect("oversized thinking markdown should be written");

    assert_eq!(
        load_qwen_thinking_channel_seed(true, &thinking_seed_file_path).await,
        None
    );
}

#[tokio::test]
async fn should_accept_thinking_markdown_at_the_exact_worker_boundary() {
    let instance_state_directory =
        tempfile::tempdir().expect("a temporary instance state directory should be created");
    let thinking_seed_file_path = instance_state_directory.path().join("thinking.md");
    std::fs::write(
        &thinking_seed_file_path,
        "R".repeat(MAX_QWEN_THINKING_CHANNEL_SEED_BYTES),
    )
    .expect("boundary-sized thinking markdown should be written");

    assert_eq!(
        load_qwen_thinking_channel_seed(true, &thinking_seed_file_path)
            .await
            .map(|thinking_seed| thinking_seed.len()),
        Some(MAX_QWEN_THINKING_CHANNEL_SEED_BYTES)
    );
}

#[tokio::test]
async fn should_treat_non_utf8_thinking_markdown_as_absent() {
    let instance_state_directory =
        tempfile::tempdir().expect("a temporary instance state directory should be created");
    let thinking_seed_file_path = instance_state_directory.path().join("thinking.md");
    std::fs::write(&thinking_seed_file_path, [0xff, 0xfe])
        .expect("non-UTF-8 thinking markdown should be written");

    assert_eq!(
        load_qwen_thinking_channel_seed(true, &thinking_seed_file_path).await,
        None
    );
}
