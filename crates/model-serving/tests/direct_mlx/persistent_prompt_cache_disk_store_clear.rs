use std::fs;
use std::future::Future;
use std::time::Duration;

use astronomical_model_serving::clear_persistent_prompt_cache_directory;

const TEST_TIMEOUT: Duration = Duration::from_secs(5);

#[tokio::test]
async fn should_clear_every_model_cache_from_the_global_root() {
    run_bounded_test(async {
        eprintln!("[cache-clear] arranging two model cache namespaces");
        let cache_root = tempfile::tempdir().expect("cache root should be created");
        create_cache_block(cache_root.path(), "publisher/model-a", "revision-a", 5);
        create_cache_block(cache_root.path(), "publisher/model-b", "revision-b", 7);

        eprintln!("[cache-clear] clearing the global cache root");
        let clear_outcome = clear_persistent_prompt_cache_directory(cache_root.path(), None)
            .expect("global cache clear should succeed");

        assert_eq!(clear_outcome.model_id, None);
        assert_eq!(clear_outcome.blocks_removed, 2);
        assert_eq!(clear_outcome.bytes_freed, 12);
        assert_directory_is_empty(cache_root.path());
    })
    .await;
}

#[tokio::test]
async fn should_clear_only_the_requested_model_and_preserve_other_models() {
    run_bounded_test(async {
        eprintln!("[cache-clear] arranging scoped model cache namespaces");
        let cache_root = tempfile::tempdir().expect("cache root should be created");
        create_cache_block(cache_root.path(), "publisher/model-a", "revision-a", 5);
        create_cache_block(cache_root.path(), "publisher/model-b", "revision-b", 7);

        let clear_outcome =
            clear_persistent_prompt_cache_directory(cache_root.path(), Some("publisher/model-a"))
                .expect("model-scoped cache clear should succeed");

        assert_eq!(clear_outcome.model_id.as_deref(), Some("publisher/model-a"));
        assert_eq!(clear_outcome.blocks_removed, 1);
        assert_eq!(clear_outcome.bytes_freed, 5);
        assert!(!cache_root.path().join("publisher/model-a").exists());
        assert!(cache_root.path().join("publisher/model-b").exists());
    })
    .await;
}

#[tokio::test]
async fn should_treat_a_missing_model_cache_as_an_idempotent_clear() {
    run_bounded_test(async {
        let cache_root = tempfile::tempdir().expect("cache root should be created");

        let clear_outcome = clear_persistent_prompt_cache_directory(
            cache_root.path(),
            Some("publisher/missing-model"),
        )
        .expect("missing model cache should already be clear");

        assert_eq!(clear_outcome.blocks_removed, 0);
        assert_eq!(clear_outcome.bytes_freed, 0);
    })
    .await;
}

#[tokio::test]
async fn should_reject_model_paths_that_can_escape_the_cache_root() {
    run_bounded_test(async {
        let cache_root = tempfile::tempdir().expect("cache root should be created");
        let preserved_file = cache_root.path().join("preserved.txt");
        fs::write(&preserved_file, b"preserved").expect("preserved fixture should be written");

        for unsafe_model_id in ["../outside", "publisher/../outside", "/outside", "."] {
            let clear_error =
                clear_persistent_prompt_cache_directory(cache_root.path(), Some(unsafe_model_id))
                    .expect_err("unsafe model path should be rejected");
            assert!(clear_error.to_string().contains("real directory"));
        }
        assert!(preserved_file.exists());
    })
    .await;
}

#[tokio::test]
async fn should_reject_an_intermediate_symlink_without_deleting_its_target() {
    run_bounded_test(async {
        use std::os::unix::fs::symlink;

        let cache_root = tempfile::tempdir().expect("cache root should be created");
        let external_directory = tempfile::tempdir().expect("external directory should be created");
        let external_model_directory = external_directory.path().join("model");
        fs::create_dir_all(&external_model_directory)
            .expect("external model directory should be created");
        let preserved_file = external_model_directory.join("preserved.safetensors");
        fs::write(&preserved_file, b"preserved").expect("external fixture should be written");
        symlink(
            external_directory.path(),
            cache_root.path().join("publisher"),
        )
        .expect("cache namespace symlink should be created");

        clear_persistent_prompt_cache_directory(cache_root.path(), Some("publisher/model"))
            .expect_err("cache clear should reject an intermediate symlink");

        assert!(preserved_file.exists());
    })
    .await;
}

#[tokio::test]
async fn should_reject_a_global_root_with_parent_directory_components() {
    run_bounded_test(async {
        let test_parent = tempfile::tempdir().expect("test parent should be created");
        let preserved_directory = test_parent.path().join("preserved-cache");
        fs::create_dir_all(&preserved_directory)
            .expect("preserved cache directory should be created");
        let preserved_file = preserved_directory.join("preserved.safetensors");
        fs::write(&preserved_file, b"preserved").expect("preserved fixture should be written");
        let unsafe_global_root = test_parent
            .path()
            .join("unused-component")
            .join("..")
            .join("preserved-cache");

        clear_persistent_prompt_cache_directory(&unsafe_global_root, None)
            .expect_err("global root with parent components should be rejected");

        assert!(preserved_file.exists());
    })
    .await;
}

#[tokio::test]
async fn should_unlink_a_global_cache_symlink_without_following_its_target() {
    run_bounded_test(async {
        use std::os::unix::fs::symlink;

        let cache_root = tempfile::tempdir().expect("cache root should be created");
        let external_directory = tempfile::tempdir().expect("external directory should be created");
        let preserved_file = external_directory.path().join("preserved.safetensors");
        fs::write(&preserved_file, b"preserved").expect("external fixture should be written");
        let cache_symlink = cache_root.path().join("linked-cache-entry");
        symlink(external_directory.path(), &cache_symlink)
            .expect("cache symlink should be created");

        let clear_outcome = clear_persistent_prompt_cache_directory(cache_root.path(), None)
            .expect("global clear should unlink cache-owned symlinks");

        assert!(clear_outcome.bytes_freed > 0);
        assert!(!cache_symlink.exists());
        assert!(preserved_file.exists());
    })
    .await;
}

async fn run_bounded_test(test_journey: impl Future<Output = ()>) {
    tokio::time::timeout(TEST_TIMEOUT, test_journey)
        .await
        .expect("cache-clear test should finish within five seconds");
}

fn create_cache_block(
    cache_root: &std::path::Path,
    model_id: &str,
    model_revision: &str,
    file_size_bytes: usize,
) {
    let cache_block_directory = cache_root
        .join(model_id)
        .join(model_revision)
        .join("blocks")
        .join("fictional-block-hash");
    fs::create_dir_all(&cache_block_directory).expect("cache block directory should be created");
    fs::write(
        cache_block_directory.join("sequence_state.safetensors"),
        vec![0_u8; file_size_bytes],
    )
    .expect("cache block fixture should be written");
}

fn assert_directory_is_empty(directory_path: &std::path::Path) {
    let mut directory_entries =
        fs::read_dir(directory_path).expect("cache root should remain readable after clear");
    assert!(directory_entries.next().is_none());
}
