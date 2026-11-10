use super::write_model_card;

#[test]
fn should_write_the_mlx_front_matter_into_an_artifact_without_a_card() {
    let directory = tempfile::tempdir().expect("a temporary directory should be creatable");

    let was_written = write_model_card(directory.path()).expect("the card should be written");

    let card_text = std::fs::read_to_string(directory.path().join("README.md"))
        .expect("the card should be readable");
    assert!(was_written);
    assert!(card_text.starts_with("---\n"));
    assert!(card_text.contains("library_name: mlx\n"));
    assert!(card_text.contains("pipeline_tag: text-generation\n"));
}

#[test]
fn should_never_replace_a_card_the_artifact_already_has() {
    let directory = tempfile::tempdir().expect("a temporary directory should be creatable");
    std::fs::write(directory.path().join("README.md"), "# Mine\n")
        .expect("the existing card should be writable");

    let was_written = write_model_card(directory.path()).expect("an existing card is not an error");

    let card_text = std::fs::read_to_string(directory.path().join("README.md"))
        .expect("the card should be readable");
    assert!(!was_written);
    assert_eq!(card_text, "# Mine\n");
}

#[test]
fn should_fail_when_the_artifact_directory_does_not_exist() {
    let directory = tempfile::tempdir().expect("a temporary directory should be creatable");

    let outcome = write_model_card(&directory.path().join("missing"));

    assert!(matches!(
        outcome,
        Err(crate::error::QuantizationError::ArtifactWriteFailure { .. })
    ));
}
