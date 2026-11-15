use super::DownloadQueue;
use crate::library::DownloadJobPublicErrorCode;

const SMALL_MODEL: &str = "mlx-community/Qwen3.5-0.8B-MLX-4bit";
const IMAGE_MODEL: &str = "mlx-community/flux2-klein-4b-4bit";
const LARGE_MODEL: &str = "araysuter/Ornith-1.5-122B-MLX-mixed-2bit";

#[test]
fn should_hand_out_models_in_the_order_they_were_asked_for() {
    let mut download_queue = DownloadQueue::default();

    assert_eq!(download_queue.enqueue(SMALL_MODEL), 1);
    assert_eq!(download_queue.enqueue(IMAGE_MODEL), 2);
    assert_eq!(download_queue.enqueue(LARGE_MODEL), 3);

    assert_eq!(download_queue.take_next().as_deref(), Some(SMALL_MODEL));
    assert_eq!(download_queue.take_next().as_deref(), Some(IMAGE_MODEL));
    assert_eq!(download_queue.waiting_position(LARGE_MODEL), Some(1));
}

#[test]
fn should_keep_the_place_of_a_model_asked_for_twice() {
    let mut download_queue = DownloadQueue::default();
    download_queue.enqueue(SMALL_MODEL);
    download_queue.enqueue(IMAGE_MODEL);

    assert_eq!(download_queue.enqueue(SMALL_MODEL), 1);

    assert_eq!(
        download_queue.waiting_huggingface_ids(),
        &[SMALL_MODEL, IMAGE_MODEL]
    );
}

#[test]
fn should_hand_out_nothing_from_an_empty_list() {
    let mut download_queue = DownloadQueue::default();

    assert_eq!(download_queue.take_next(), None);
    assert!(!download_queue.has_waiting_models());
    assert_eq!(download_queue.waiting_position(SMALL_MODEL), None);
}

#[test]
fn should_move_a_failed_model_back_to_waiting_when_asked_for_again() {
    let mut download_queue = DownloadQueue::default();
    download_queue.record_failure(IMAGE_MODEL, DownloadJobPublicErrorCode::DownloadFailed);

    assert_eq!(download_queue.enqueue(IMAGE_MODEL), 1);

    assert!(download_queue.failed_entries().is_empty());
    assert_eq!(download_queue.waiting_huggingface_ids(), &[IMAGE_MODEL]);
}

#[test]
fn should_keep_only_the_latest_failure_of_a_model() {
    let mut download_queue = DownloadQueue::default();

    download_queue.record_failure(IMAGE_MODEL, DownloadJobPublicErrorCode::DownloadFailed);
    download_queue.record_failure(IMAGE_MODEL, DownloadJobPublicErrorCode::InsufficientDisk);

    assert_eq!(download_queue.failed_entries().len(), 1);
    assert_eq!(
        download_queue.failed_entries()[0].error_code,
        DownloadJobPublicErrorCode::InsufficientDisk
    );
}

#[test]
fn should_remove_a_waiting_or_failed_model_and_report_an_unlisted_one() {
    let mut download_queue = DownloadQueue::default();
    download_queue.enqueue(SMALL_MODEL);
    download_queue.record_failure(IMAGE_MODEL, DownloadJobPublicErrorCode::DownloadGated);

    assert!(download_queue.remove(SMALL_MODEL));
    assert!(download_queue.remove(IMAGE_MODEL));
    assert!(!download_queue.remove(LARGE_MODEL));

    assert_eq!(download_queue, DownloadQueue::default());
}

#[test]
fn should_round_trip_through_its_saved_form() {
    let mut download_queue = DownloadQueue::default();
    download_queue.enqueue(SMALL_MODEL);
    download_queue.record_failure(IMAGE_MODEL, DownloadJobPublicErrorCode::ChecksumMismatch);

    let saved_text = serde_json::to_string(&download_queue).expect("the list serializes");
    let restored_queue: DownloadQueue =
        serde_json::from_str(&saved_text).expect("the saved list parses");

    assert_eq!(restored_queue, download_queue);
}

#[test]
fn should_refuse_a_saved_list_with_unknown_fields() {
    let saved_text = r#"{"waiting_huggingface_ids":[],"failed_entries":[],"priority":1}"#;

    assert!(serde_json::from_str::<DownloadQueue>(saved_text).is_err());
}
