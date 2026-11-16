use super::DownloadQueueStore;
use crate::library::DownloadJobPublicErrorCode;
use crate::library::download::queue::download_queue::DownloadQueue;

const SMALL_MODEL: &str = "mlx-community/Qwen3.5-0.8B-MLX-4bit";
const IMAGE_MODEL: &str = "mlx-community/flux2-klein-4b-4bit";

fn models_folder() -> tempfile::TempDir {
    tempfile::tempdir().expect("a temporary models folder")
}

#[test]
fn should_load_the_list_it_saved() {
    let models_folder = models_folder();
    let queue_store = DownloadQueueStore::new(models_folder.path().to_path_buf());
    let mut download_queue = DownloadQueue::default();
    download_queue.enqueue(SMALL_MODEL);
    download_queue.record_failure(IMAGE_MODEL, DownloadJobPublicErrorCode::DownloadFailed);

    queue_store.save(&download_queue).expect("the list saves");

    assert_eq!(queue_store.load(), download_queue);
}

#[test]
fn should_load_an_empty_list_when_none_was_saved() {
    let models_folder = models_folder();
    let queue_store = DownloadQueueStore::new(models_folder.path().to_path_buf());

    assert_eq!(queue_store.load(), DownloadQueue::default());
}

#[test]
fn should_remove_the_file_when_the_list_becomes_empty() {
    let models_folder = models_folder();
    let queue_store = DownloadQueueStore::new(models_folder.path().to_path_buf());
    let mut download_queue = DownloadQueue::default();
    download_queue.enqueue(SMALL_MODEL);
    queue_store.save(&download_queue).expect("the list saves");

    queue_store
        .save(&DownloadQueue::default())
        .expect("the empty list saves");

    assert!(!queue_store.queue_file_path().exists());
}

#[test]
fn should_start_an_empty_list_from_an_unreadable_file() {
    let models_folder = models_folder();
    let queue_store = DownloadQueueStore::new(models_folder.path().to_path_buf());
    std::fs::write(queue_store.queue_file_path(), "not a list").expect("the bad file writes");

    assert_eq!(queue_store.load(), DownloadQueue::default());
}

#[test]
fn should_create_a_missing_models_folder_when_saving() {
    let models_folder = models_folder();
    let missing_models_folder = models_folder.path().join("models");
    let queue_store = DownloadQueueStore::new(missing_models_folder);
    let mut download_queue = DownloadQueue::default();
    download_queue.enqueue(SMALL_MODEL);

    queue_store
        .save(&download_queue)
        .expect("the list saves into a new folder");

    assert_eq!(queue_store.load(), download_queue);
}

#[test]
fn should_start_an_empty_list_from_a_file_larger_than_any_list_it_writes() {
    let models_folder = models_folder();
    let queue_store = DownloadQueueStore::new(models_folder.path().to_path_buf());
    std::fs::write(queue_store.queue_file_path(), vec![b' '; 1_000_001])
        .expect("the oversized file writes");

    assert_eq!(queue_store.load(), DownloadQueue::default());
}

#[cfg(unix)]
#[test]
fn should_refuse_to_write_through_a_link_planted_at_the_temporary_name() {
    let models_folder = models_folder();
    let queue_store = DownloadQueueStore::new(models_folder.path().to_path_buf());
    let link_target = models_folder.path().join("elsewhere.txt");
    std::fs::write(&link_target, "untouched").expect("the link target writes");
    std::os::unix::fs::symlink(
        &link_target,
        models_folder.path().join(".download-queue.json.tmp"),
    )
    .expect("the planted link is created");
    let mut download_queue = DownloadQueue::default();
    download_queue.enqueue(SMALL_MODEL);

    assert!(queue_store.save(&download_queue).is_err());
    assert_eq!(
        std::fs::read_to_string(&link_target).expect("the link target reads"),
        "untouched"
    );
}
