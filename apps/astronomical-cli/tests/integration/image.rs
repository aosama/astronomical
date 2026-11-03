//! The `astronomical image` one-shot journey: a prompt in, one PNG on disk,
//! the saved path on stdout, process exits. The daemon side is a stub
//! speaking the real framed protocol over a real unix socket.

use std::{
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::Duration,
};

use astronomical_cli::{ImageArguments, ImageDependencies, ImageError};
use astronomical_config::AstronomicalRuntimeInstance;
use astronomical_ipc_protocol::{DaemonRequest, ImageGenerationFailureReason};
use tokio::time::timeout;

use crate::integration::support::environment::{
    DOWNLOAD_POLL_INTERVAL, SOCKET_FILE_NAME, TEST_TIMEOUT, fresh_test_directory,
};
use crate::integration::support::stub_daemon::{
    StubCatalogEntry, StubDaemonConfig, StubImageOutcome, StubInstalledModel, spawn_stub_daemon,
    stub_download_job,
};

/// The built-in image model's id, which an unflagged run must ask for.
/// Read from the config crate, not retyped, so changing the default cannot
/// silently break these journeys.
const BUILTIN_IMAGE_MODEL_ID: &str = astronomical_config::BUILTIN_DEFAULT_IMAGE_MODEL_ID;
const CHAT_MODEL_ID: &str = "test/local-chatter";

/// A PNG signature and a few bytes. The CLI checks only the signature: the
/// daemon has already decoded the whole image before it sends it.
const PNG_BYTES: [u8; 12] = [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A, 0, 0, 0, 0];

fn image_arguments(model_id: Option<&str>, output_path: Option<PathBuf>) -> ImageArguments {
    ImageArguments {
        prompt: "a lighthouse at dusk".to_owned(),
        width_pixels: 1_024,
        height_pixels: 576,
        model_id: model_id.map(str::to_owned),
        seed: Some(42),
        output_path,
    }
}

/// A daemon with the built-in image model and a chat model installed, the
/// chat model being the configured default: the situation an unflagged
/// `image` must not mistake for an image default.
fn image_stub_config(image_outcome: StubImageOutcome) -> StubDaemonConfig {
    StubDaemonConfig {
        installed_models: vec![
            StubInstalledModel::image(BUILTIN_IMAGE_MODEL_ID, false),
            StubInstalledModel::chat(CHAT_MODEL_ID, true),
        ],
        default_model_id: Some(CHAT_MODEL_ID.to_owned()),
        image_outcome,
        ..Default::default()
    }
}

/// Everything one journey produced.
struct ImageRun {
    outcome: Result<(), ImageError>,
    stdout: String,
    stderr: String,
    captured_request: Option<DaemonRequest>,
    test_directory: PathBuf,
}

struct JourneyBounds {
    generation_bound: Duration,
    progress_interval: Duration,
}

impl JourneyBounds {
    fn patient() -> Self {
        JourneyBounds {
            generation_bound: TEST_TIMEOUT,
            progress_interval: Duration::from_secs(1),
        }
    }
}

/// Starts the stub daemon when the test wants one, and hands back the shared
/// capture of the image request it will see.
fn start_stub(
    socket_path: &Path,
    stub_config: Option<StubDaemonConfig>,
) -> (
    Option<tokio::task::JoinHandle<()>>,
    Arc<Mutex<Option<DaemonRequest>>>,
) {
    match stub_config {
        Some(stub_config) => {
            let request_capture = Arc::clone(&stub_config.image_request_capture);
            (
                Some(spawn_stub_daemon(socket_path.to_path_buf(), stub_config)),
                request_capture,
            )
        }
        None => (None, Arc::new(Mutex::new(None))),
    }
}

/// Runs the journey against a stub daemon (or none, for the not-running case).
async fn run_journey(
    test_name: &str,
    stub_config: Option<StubDaemonConfig>,
    image_arguments: ImageArguments,
    journey_bounds: JourneyBounds,
) -> ImageRun {
    let test_directory = fresh_test_directory("image", test_name);
    let socket_path = test_directory.join(SOCKET_FILE_NAME);
    let (stub_daemon_task, request_capture) = start_stub(&socket_path, stub_config);
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let mut image_dependencies = ImageDependencies {
        candidate_socket_paths: vec![socket_path],
        runtime_instance: AstronomicalRuntimeInstance::Development,
        default_output_directory: test_directory.clone(),
        stdout: &mut stdout,
        stderr: &mut stderr,
        request_timeout: TEST_TIMEOUT,
        generation_bound: journey_bounds.generation_bound,
        progress_interval: journey_bounds.progress_interval,
        download_stall_bound: TEST_TIMEOUT,
        download_poll_interval: DOWNLOAD_POLL_INTERVAL,
    };
    let outcome = timeout(
        TEST_TIMEOUT,
        astronomical_cli::run_image(&image_arguments, &mut image_dependencies),
    )
    .await
    .expect("the image journey should finish inside the test timeout");
    if let Some(stub_daemon_task) = stub_daemon_task {
        stub_daemon_task.abort();
    }
    let captured_request = request_capture
        .lock()
        .expect("the image request capture should stay live")
        .clone();
    ImageRun {
        outcome,
        stdout: String::from_utf8_lossy(&stdout).into_owned(),
        stderr: String::from_utf8_lossy(&stderr).into_owned(),
        captured_request,
        test_directory,
    }
}

/// Fails the test with the journey's own error when it did not succeed.
fn assert_succeeded(image_run: &ImageRun, expectation: &str) {
    if let Err(image_error) = &image_run.outcome {
        panic!("{expectation}: {image_error}");
    }
}

fn captured_image_request(image_run: &ImageRun) -> (String, String, u32, u32, Option<u64>) {
    let Some(DaemonRequest::ImageGenerate(image_request)) = image_run.captured_request.clone()
    else {
        panic!("the daemon should have received an image request");
    };
    (
        image_request.model,
        image_request.prompt,
        image_request.width_pixels,
        image_request.height_pixels,
        image_request.seed,
    )
}

fn directory_entry_count(directory: &Path) -> usize {
    let mut entry_count = 0;
    for directory_entry in
        std::fs::read_dir(directory).expect("the test directory should be readable")
    {
        let directory_entry = directory_entry.expect("a directory entry should be readable");
        if directory_entry.file_name() != SOCKET_FILE_NAME {
            entry_count += 1;
        }
    }
    entry_count
}

/// What a script and a person each see after a saved image: exactly one
/// absolute path on stdout, and the seed and size on stderr.
fn assert_saved_output_report(image_run: &ImageRun) {
    let printed_path = image_run.stdout.trim_end();
    assert!(printed_path.ends_with("lighthouse.png"), "{printed_path}");
    assert!(
        Path::new(printed_path).is_absolute(),
        "stdout carries only the absolute path: {printed_path}"
    );
    assert_eq!(
        image_run.stdout.lines().count(),
        1,
        "stdout must hold exactly the path for scripts"
    );
    for expected_text in ["seed 42", "--seed 42", "1024x576"] {
        assert!(
            image_run.stderr.contains(expected_text),
            "{}",
            image_run.stderr
        );
    }
}

#[tokio::test]
async fn should_save_the_png_print_the_path_and_report_the_seed() {
    let output_directory = fresh_test_directory("image", "saved-output");
    let output_path = output_directory.join("lighthouse.png");
    let image_run = run_journey(
        "saved",
        Some(image_stub_config(StubImageOutcome::Completed(
            PNG_BYTES.to_vec(),
        ))),
        image_arguments(None, Some(output_path.clone())),
        JourneyBounds::patient(),
    )
    .await;

    assert_succeeded(&image_run, "a completed image must succeed");
    assert_eq!(
        std::fs::read(&output_path).expect("the PNG should be saved"),
        PNG_BYTES
    );
    assert_saved_output_report(&image_run);
    let _ = std::fs::remove_dir_all(&output_directory);
    let _ = std::fs::remove_dir_all(&image_run.test_directory);
}

#[tokio::test]
async fn should_send_exactly_what_the_user_chose() {
    let image_run = run_journey(
        "request-shape",
        Some(image_stub_config(StubImageOutcome::Completed(
            PNG_BYTES.to_vec(),
        ))),
        image_arguments(None, None),
        JourneyBounds::patient(),
    )
    .await;

    assert_succeeded(&image_run, "a completed image must succeed");
    let (model, prompt, width_pixels, height_pixels, seed) = captured_image_request(&image_run);
    assert_eq!(model, BUILTIN_IMAGE_MODEL_ID);
    assert_eq!(prompt, "a lighthouse at dusk");
    assert_eq!((width_pixels, height_pixels), (1_024, 576));
    assert_eq!(seed, Some(42));
    let _ = std::fs::remove_dir_all(&image_run.test_directory);
}

#[tokio::test]
async fn should_use_the_builtin_image_model_not_the_chat_default() {
    let image_run = run_journey(
        "builtin-model",
        Some(image_stub_config(StubImageOutcome::Completed(
            PNG_BYTES.to_vec(),
        ))),
        image_arguments(None, None),
        JourneyBounds::patient(),
    )
    .await;

    assert_succeeded(
        &image_run,
        "an unflagged image must reach the built-in image model",
    );
    let (model, ..) = captured_image_request(&image_run);
    assert_eq!(
        model, BUILTIN_IMAGE_MODEL_ID,
        "the chat default cannot draw and must be skipped"
    );
    let _ = std::fs::remove_dir_all(&image_run.test_directory);
}

#[tokio::test]
async fn should_name_the_file_in_the_default_directory_when_no_output_is_given() {
    let image_run = run_journey(
        "default-name",
        Some(image_stub_config(StubImageOutcome::Completed(
            PNG_BYTES.to_vec(),
        ))),
        image_arguments(None, None),
        JourneyBounds::patient(),
    )
    .await;

    assert_succeeded(&image_run, "a completed image must succeed");
    let printed_path = PathBuf::from(image_run.stdout.trim_end());
    let file_name = printed_path
        .file_name()
        .and_then(std::ffi::OsStr::to_str)
        .expect("a file name");
    assert!(
        file_name.starts_with("astronomical-image-") && file_name.ends_with(".png"),
        "{file_name}"
    );
    assert!(printed_path.is_file());
    let _ = std::fs::remove_dir_all(&image_run.test_directory);
}

#[tokio::test]
async fn should_refuse_an_existing_output_without_touching_it_or_the_daemon() {
    let existing_directory = fresh_test_directory("image", "existing");
    let existing_path = existing_directory.join("taken.png");
    std::fs::write(&existing_path, b"precious").expect("the existing file should be writable");

    let image_run = run_journey(
        "existing-out",
        Some(image_stub_config(StubImageOutcome::Completed(
            PNG_BYTES.to_vec(),
        ))),
        image_arguments(None, Some(existing_path.clone())),
        JourneyBounds::patient(),
    )
    .await;

    let Err(image_error) = &image_run.outcome else {
        panic!("an existing output must be refused");
    };
    assert!(
        matches!(image_error, ImageError::OutputExists { .. }),
        "{image_error:?}"
    );
    assert!(image_error.is_usage_error());
    assert_eq!(
        std::fs::read(&existing_path).expect("still there"),
        b"precious"
    );
    assert!(
        image_run.captured_request.is_none(),
        "no GPU work for an output that cannot be written"
    );
    let _ = std::fs::remove_dir_all(&existing_directory);
    let _ = std::fs::remove_dir_all(&image_run.test_directory);
}

#[tokio::test]
async fn should_refuse_an_output_that_is_not_a_png_name() {
    let output_directory = fresh_test_directory("image", "not-png");
    let jpeg_path = output_directory.join("lighthouse.jpg");

    let image_run = run_journey(
        "not-png",
        Some(image_stub_config(StubImageOutcome::Completed(
            PNG_BYTES.to_vec(),
        ))),
        image_arguments(None, Some(jpeg_path.clone())),
        JourneyBounds::patient(),
    )
    .await;

    let Err(image_error) = &image_run.outcome else {
        panic!("a mislabelled output must be refused");
    };
    assert!(
        matches!(image_error, ImageError::OutputMustBePng { .. }),
        "{image_error:?}"
    );
    assert!(image_error.is_usage_error());
    assert!(!jpeg_path.exists());
    assert!(image_run.captured_request.is_none());
    let _ = std::fs::remove_dir_all(&output_directory);
    let _ = std::fs::remove_dir_all(&image_run.test_directory);
}

#[tokio::test]
async fn should_accept_an_uppercase_png_extension() {
    let output_directory = fresh_test_directory("image", "upper-png");
    let uppercase_path = output_directory.join("LIGHTHOUSE.PNG");

    let image_run = run_journey(
        "upper-png",
        Some(image_stub_config(StubImageOutcome::Completed(
            PNG_BYTES.to_vec(),
        ))),
        image_arguments(None, Some(uppercase_path.clone())),
        JourneyBounds::patient(),
    )
    .await;

    assert_succeeded(&image_run, "an uppercase .PNG is still a PNG name");
    assert!(uppercase_path.is_file());
    let _ = std::fs::remove_dir_all(&output_directory);
    let _ = std::fs::remove_dir_all(&image_run.test_directory);
}

#[tokio::test]
async fn should_fail_fast_when_the_output_folder_does_not_exist() {
    let missing_folder_path = fresh_test_directory("image", "missing-folder")
        .join("no-such-folder")
        .join("lighthouse.png");

    let image_run = run_journey(
        "missing-folder",
        Some(image_stub_config(StubImageOutcome::Completed(
            PNG_BYTES.to_vec(),
        ))),
        image_arguments(None, Some(missing_folder_path)),
        JourneyBounds::patient(),
    )
    .await;

    let Err(image_error) = &image_run.outcome else {
        panic!("an unwritable output must be refused before any generation");
    };
    assert!(
        matches!(image_error, ImageError::OutputUnwritable { .. }),
        "{image_error:?}"
    );
    assert!(image_run.captured_request.is_none());
    let _ = std::fs::remove_dir_all(&image_run.test_directory);
}

#[tokio::test]
async fn should_leave_no_empty_file_when_the_daemon_rejects_the_request() {
    let image_run = run_journey(
        "rejected",
        Some(image_stub_config(StubImageOutcome::Rejected(
            "width must be a multiple of 16 pixels for FLUX.2-klein-4B, received 1000".to_owned(),
        ))),
        image_arguments(None, None),
        JourneyBounds::patient(),
    )
    .await;

    let Err(image_error) = &image_run.outcome else {
        panic!("a rejected request must fail");
    };
    assert!(
        matches!(image_error, ImageError::ImageRejected { .. }),
        "{image_error:?}"
    );
    assert!(
        image_error.to_string().contains("multiple of 16"),
        "{image_error}"
    );
    assert!(
        !image_error.is_usage_error(),
        "the daemon's verdict exits 1, not 2"
    );
    assert_eq!(
        directory_entry_count(&image_run.test_directory),
        0,
        "the reserved file must be removed"
    );
    assert!(image_run.stdout.is_empty());
    let _ = std::fs::remove_dir_all(&image_run.test_directory);
}

#[tokio::test]
async fn should_leave_no_empty_file_when_the_worker_fails_the_image() {
    let image_run = run_journey(
        "worker-failed",
        Some(image_stub_config(StubImageOutcome::Failed(
            ImageGenerationFailureReason::EngineBusy,
        ))),
        image_arguments(None, None),
        JourneyBounds::patient(),
    )
    .await;

    let Err(image_error) = &image_run.outcome else {
        panic!("a worker failure must fail the run");
    };
    assert!(image_error.to_string().contains("busy"), "{image_error}");
    assert_eq!(directory_entry_count(&image_run.test_directory), 0);
    let _ = std::fs::remove_dir_all(&image_run.test_directory);
}

#[tokio::test]
async fn should_refuse_a_reply_that_is_not_a_png_and_save_nothing() {
    let image_run = run_journey(
        "not-a-png",
        Some(image_stub_config(StubImageOutcome::Completed(
            b"definitely not a png".to_vec(),
        ))),
        image_arguments(None, None),
        JourneyBounds::patient(),
    )
    .await;

    let Err(image_error) = &image_run.outcome else {
        panic!("a reply that is not a PNG must not be saved");
    };
    assert!(
        matches!(image_error, ImageError::ReturnedImageInvalid),
        "{image_error:?}"
    );
    assert_eq!(directory_entry_count(&image_run.test_directory), 0);
    assert!(
        image_run.stdout.is_empty(),
        "no path may be printed for an image that was not saved"
    );
    let _ = std::fs::remove_dir_all(&image_run.test_directory);
}

#[tokio::test]
async fn should_refuse_a_chat_model_as_not_an_image_model() {
    let image_run = run_journey(
        "chat-model",
        Some(image_stub_config(StubImageOutcome::Completed(
            PNG_BYTES.to_vec(),
        ))),
        image_arguments(Some(CHAT_MODEL_ID), None),
        JourneyBounds::patient(),
    )
    .await;

    let Err(image_error) = &image_run.outcome else {
        panic!("a chat model cannot draw");
    };
    assert!(
        matches!(image_error, ImageError::ModelUnavailable { .. }),
        "{image_error:?}"
    );
    assert!(
        image_error.to_string().contains("image-generation model"),
        "{image_error}"
    );
    assert!(image_error.is_usage_error());
    assert!(image_run.captured_request.is_none());
    assert_eq!(directory_entry_count(&image_run.test_directory), 0);
    let _ = std::fs::remove_dir_all(&image_run.test_directory);
}

/// A Mac with only a chat model installed, whose catalog offers the image
/// model as a download that finishes after one poll.
fn missing_image_model_stub_config() -> StubDaemonConfig {
    let huggingface_id_text = format!("test/{BUILTIN_IMAGE_MODEL_ID}");
    let huggingface_id = huggingface_id_text.as_str();
    let download_job = stub_download_job(
        huggingface_id,
        "downloading",
        1_000_000_000,
        2_000_000_000,
        None,
    );
    StubDaemonConfig {
        installed_models: vec![StubInstalledModel::chat(CHAT_MODEL_ID, true)],
        default_model_id: Some(CHAT_MODEL_ID.to_owned()),
        catalog_entries: vec![StubCatalogEntry::image(
            huggingface_id,
            BUILTIN_IMAGE_MODEL_ID,
            false,
        )],
        catalog_entries_after_download: Some(vec![StubCatalogEntry::image(
            huggingface_id,
            BUILTIN_IMAGE_MODEL_ID,
            true,
        )]),
        download_jobs: vec![Some(download_job), None],
        image_outcome: StubImageOutcome::Completed(PNG_BYTES.to_vec()),
        ..Default::default()
    }
}

#[tokio::test]
async fn should_download_the_image_model_when_the_mac_lacks_it() {
    let image_run = run_journey(
        "download",
        Some(missing_image_model_stub_config()),
        image_arguments(None, None),
        JourneyBounds::patient(),
    )
    .await;

    assert_succeeded(
        &image_run,
        "an auto-download must complete before the image",
    );
    assert!(
        image_run.stderr.contains("0.93 GB / 1.86 GB"),
        "{}",
        image_run.stderr
    );
    let (model, ..) = captured_image_request(&image_run);
    assert_eq!(model, BUILTIN_IMAGE_MODEL_ID);
    let _ = std::fs::remove_dir_all(&image_run.test_directory);
}

#[tokio::test]
async fn should_report_progress_then_give_up_when_the_worker_never_answers() {
    let image_run = run_journey(
        "stalled",
        Some(image_stub_config(StubImageOutcome::Stalls)),
        image_arguments(None, None),
        JourneyBounds {
            generation_bound: Duration::from_secs(1),
            progress_interval: Duration::from_millis(200),
        },
    )
    .await;

    let Err(image_error) = &image_run.outcome else {
        panic!("a wedged worker must not hang the CLI");
    };
    assert!(
        matches!(image_error, ImageError::GenerationTimedOut { seconds: 1 }),
        "{image_error:?}"
    );
    assert!(
        image_run.stderr.contains("generating:"),
        "progress lines were expected: {}",
        image_run.stderr
    );
    assert!(
        image_run.stderr.contains("s elapsed"),
        "{}",
        image_run.stderr
    );
    assert_eq!(
        directory_entry_count(&image_run.test_directory),
        0,
        "the reserved file must be removed"
    );
    let _ = std::fs::remove_dir_all(&image_run.test_directory);
}

#[tokio::test]
async fn should_report_a_missing_daemon_as_not_running_and_leave_no_file() {
    let image_run = run_journey(
        "not-running",
        None,
        image_arguments(None, None),
        JourneyBounds::patient(),
    )
    .await;

    let Err(image_error) = &image_run.outcome else {
        panic!("a missing daemon must fail");
    };
    assert!(
        matches!(image_error, ImageError::DaemonNotRunning { .. }),
        "{image_error:?}"
    );
    assert!(
        image_error
            .to_string()
            .contains("Astronomical (development channel) isn't running"),
        "{image_error}"
    );
    assert_eq!(directory_entry_count(&image_run.test_directory), 0);
    let _ = std::fs::remove_dir_all(&image_run.test_directory);
}
