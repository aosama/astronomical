use super::{PROBE_SOURCE_VIEW_DIRECTORY_NAME, loadable_model_directory};

/// A safetensors file holding one two-element bfloat16 tensor.
fn write_single_tensor_shard(shard_path: &std::path::Path, tensor_name: &str) {
    let header =
        format!(r#"{{"{tensor_name}":{{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}}}}"#);
    let mut shard_bytes = (header.len() as u64).to_le_bytes().to_vec();
    shard_bytes.extend_from_slice(header.as_bytes());
    shard_bytes.extend_from_slice(&[0_u8; 4]);
    std::fs::write(shard_path, shard_bytes).expect("the shard should be writable");
}

#[test]
fn should_return_the_source_itself_when_it_already_has_an_index() {
    let source = tempfile::tempdir().expect("a temporary directory should be creatable");
    let output = tempfile::tempdir().expect("a temporary directory should be creatable");
    std::fs::write(source.path().join("model.safetensors.index.json"), "{}")
        .expect("the index should be writable");

    let (probe_directory, was_built) = loadable_model_directory(source.path(), output.path())
        .expect("an indexed source needs no view");

    assert_eq!(probe_directory, source.path());
    assert!(!was_built);
    assert!(
        !output
            .path()
            .join(PROBE_SOURCE_VIEW_DIRECTORY_NAME)
            .exists()
    );
}

#[test]
fn should_build_a_linked_view_with_the_index_a_single_file_source_lacks() {
    let source = tempfile::tempdir().expect("a temporary directory should be creatable");
    let output = tempfile::tempdir().expect("a temporary directory should be creatable");
    write_single_tensor_shard(
        &source.path().join("model.safetensors"),
        "model.norm.weight",
    );
    std::fs::write(source.path().join("config.json"), "{}").expect("the config is writable");

    let (probe_directory, was_built) =
        loadable_model_directory(source.path(), output.path()).expect("the view should be built");

    assert!(was_built);
    let linked_shard = probe_directory
        .join("model.safetensors")
        .symlink_metadata()
        .expect("the shard link exists");
    assert!(linked_shard.file_type().is_file());
    assert_eq!(std::os::unix::fs::MetadataExt::nlink(&linked_shard), 2);
    assert!(probe_directory.join("config.json").is_file());
    let index: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(probe_directory.join("model.safetensors.index.json"))
            .expect("the index should be readable"),
    )
    .expect("the index should parse");
    assert_eq!(
        index["weight_map"]["model.norm.weight"],
        "model.safetensors"
    );
    assert_eq!(index["metadata"]["total_size"], 4);
}

#[test]
fn should_replace_a_view_left_by_an_interrupted_run() {
    let source = tempfile::tempdir().expect("a temporary directory should be creatable");
    let output = tempfile::tempdir().expect("a temporary directory should be creatable");
    write_single_tensor_shard(
        &source.path().join("model.safetensors"),
        "model.norm.weight",
    );
    let stale_view = output.path().join(PROBE_SOURCE_VIEW_DIRECTORY_NAME);
    std::fs::create_dir_all(&stale_view).expect("the stale view should be creatable");
    std::fs::write(stale_view.join("leftover.txt"), "stale").expect("the leftover is writable");

    let (probe_directory, _) =
        loadable_model_directory(source.path(), output.path()).expect("the view should be rebuilt");

    assert!(!probe_directory.join("leftover.txt").exists());
}

#[test]
fn should_fail_for_a_source_without_any_shard() {
    let source = tempfile::tempdir().expect("a temporary directory should be creatable");
    let output = tempfile::tempdir().expect("a temporary directory should be creatable");

    let outcome = loadable_model_directory(source.path(), output.path());

    assert!(matches!(
        outcome,
        Err(crate::error::QuantizationError::NoSafetensorsShards { .. })
    ));
}

#[test]
fn should_build_a_view_of_plain_files_for_a_source_stored_as_symbolic_links() {
    // The Hugging Face cache layout: every file of a snapshot is a link
    // into a blob store, index included.
    let blobs = tempfile::tempdir().expect("a temporary directory should be creatable");
    let source = tempfile::tempdir().expect("a temporary directory should be creatable");
    let output = tempfile::tempdir().expect("a temporary directory should be creatable");
    write_single_tensor_shard(&blobs.path().join("shard-blob"), "model.norm.weight");
    std::fs::write(blobs.path().join("index-blob"), r#"{"weight_map": {}}"#)
        .expect("the index blob should be writable");
    std::os::unix::fs::symlink(
        blobs.path().join("shard-blob"),
        source.path().join("model.safetensors"),
    )
    .expect("the shard link should be creatable");
    std::os::unix::fs::symlink(
        blobs.path().join("index-blob"),
        source.path().join("model.safetensors.index.json"),
    )
    .expect("the index link should be creatable");

    let (probe_directory, was_built) =
        loadable_model_directory(source.path(), output.path()).expect("the view should be built");

    assert!(was_built);
    for file_name in ["model.safetensors", "model.safetensors.index.json"] {
        let viewed_file = probe_directory
            .join(file_name)
            .symlink_metadata()
            .expect("the viewed file exists");
        assert!(
            viewed_file.file_type().is_file(),
            "{file_name} is still a link"
        );
    }
    // The source's own index is kept, not regenerated.
    assert_eq!(
        std::fs::read_to_string(probe_directory.join("model.safetensors.index.json"))
            .expect("the index should be readable"),
        r#"{"weight_map": {}}"#
    );
}
