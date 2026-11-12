//! The `astronomical speak` one-shot journey: text in, one audio file on
//! disk, the saved path on stdout, process exits. The daemon side is a stub
//! speaking the real framed protocol over a real unix socket.

use std::path::PathBuf;

use astronomical_cli::{SpeakArguments, SpeakDependencies, SpeakError};
use astronomical_config::AstronomicalRuntimeInstance;
use astronomical_ipc_protocol::DaemonRequest;
use astronomical_speech_contract::SpeechFailure;
use tokio::time::timeout;

use crate::integration::support::environment::{
    DOWNLOAD_POLL_INTERVAL, SOCKET_FILE_NAME, TEST_TIMEOUT, fresh_test_directory,
};
use crate::integration::support::stub_daemon::{
    StubDaemonConfig, StubInstalledModel, StubSpeechOutcome, spawn_stub_daemon,
};

/// The built-in speech model's id, which an unflagged run must ask for.
const BUILTIN_SPEECH_MODEL_ID: &str = astronomical_config::BUILTIN_DEFAULT_SPEECH_MODEL_ID;
const CHAT_MODEL_ID: &str = "test/local-chatter";
const ROMEO_LINE: &str = "But soft, what light through yonder window breaks?";

fn speak_arguments(output_path: Option<PathBuf>) -> SpeakArguments {
    SpeakArguments {
        text: Some(ROMEO_LINE.to_owned()),
        text_file: None,
        model_id: None,
        voice_name: Some("am_michael".to_owned()),
        speed: 1.25,
        seed: Some(42),
        output_path,
    }
}

/// A daemon with the built-in speech model and a resident chat model that is
/// the configured default: an unflagged `speak` must not mistake it for a
/// speech model.
fn speech_stub_config(speech_outcome: StubSpeechOutcome) -> StubDaemonConfig {
    StubDaemonConfig {
        installed_models: vec![
            StubInstalledModel::speech(BUILTIN_SPEECH_MODEL_ID, false),
            StubInstalledModel::chat(CHAT_MODEL_ID, true),
        ],
        default_model_id: Some(CHAT_MODEL_ID.to_owned()),
        speech_outcome,
        ..Default::default()
    }
}

struct SpeakRun {
    outcome: Result<(), SpeakError>,
    stdout: String,
    captured_request: Option<DaemonRequest>,
    test_directory: PathBuf,
}

async fn run_journey(
    test_name: &str,
    stub_config: StubDaemonConfig,
    speak_arguments: SpeakArguments,
) -> SpeakRun {
    let test_directory = fresh_test_directory("speak", test_name);
    let socket_path = test_directory.join(SOCKET_FILE_NAME);
    let request_capture = std::sync::Arc::clone(&stub_config.speak_request_capture);
    let stub_daemon_task = spawn_stub_daemon(socket_path.clone(), stub_config);
    let mut unused_stdin: &[u8] = b"";
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let mut speak_dependencies = SpeakDependencies {
        candidate_socket_paths: vec![socket_path],
        runtime_instance: AstronomicalRuntimeInstance::Development,
        default_output_directory: test_directory.clone(),
        stdin: &mut unused_stdin,
        stdout: &mut stdout,
        stderr: &mut stderr,
        request_timeout: TEST_TIMEOUT,
        generation_bound: TEST_TIMEOUT,
        progress_interval: std::time::Duration::from_secs(1),
        download_stall_bound: TEST_TIMEOUT,
        download_poll_interval: DOWNLOAD_POLL_INTERVAL,
    };
    let outcome = timeout(
        TEST_TIMEOUT,
        astronomical_cli::run_speak(&speak_arguments, &mut speak_dependencies),
    )
    .await
    .expect("the speak journey should finish inside the test timeout");
    stub_daemon_task.abort();
    let captured_request = request_capture
        .lock()
        .expect("the speak request capture should stay live")
        .clone();
    SpeakRun {
        outcome,
        stdout: String::from_utf8_lossy(&stdout).into_owned(),
        captured_request,
        test_directory,
    }
}

// Positive: an unflagged run asks the built-in speech model, carries every
// flag across the socket, joins the audio frames, and saves a WAV whose path
// is the only thing on stdout.
#[tokio::test]
async fn should_speak_with_the_built_in_model_and_save_a_wav() {
    let speak_run = run_journey(
        "built_in_wav",
        speech_stub_config(StubSpeechOutcome::Completed(vec![1, 0, 2, 0, 3, 0, 4, 0])),
        speak_arguments(None),
    )
    .await;

    if let Err(speak_error) = &speak_run.outcome {
        panic!("the journey should succeed: {speak_error}");
    }
    let Some(DaemonRequest::SpeakGenerate(speak_request)) = speak_run.captured_request else {
        panic!("the daemon should have received a speak request");
    };
    assert_eq!(
        speak_request.model.as_deref(),
        Some(BUILTIN_SPEECH_MODEL_ID)
    );
    assert_eq!(speak_request.text, ROMEO_LINE);
    assert_eq!(speak_request.voice_name.as_deref(), Some("am_michael"));
    assert_eq!(speak_request.speed, 1.25);
    assert_eq!(speak_request.seed, Some(42));
    let saved_path = PathBuf::from(speak_run.stdout.trim());
    let saved_bytes = std::fs::read(&saved_path).expect("the WAV should be on disk");
    assert_eq!(&saved_bytes[..4], b"RIFF");
    assert_eq!(&saved_bytes[44..], [1, 0, 2, 0, 3, 0, 4, 0]);
}

// Edge: a .pcm output is the bare samples, both frames joined in order.
#[tokio::test]
async fn should_save_bare_samples_for_a_pcm_output() {
    let test_directory = fresh_test_directory("speak", "pcm_target");
    let pcm_path = test_directory.join("romeo.pcm");
    let speak_run = run_journey(
        "pcm_output",
        speech_stub_config(StubSpeechOutcome::Completed(vec![5, 0, 6, 0])),
        speak_arguments(Some(pcm_path.clone())),
    )
    .await;

    assert!(speak_run.outcome.is_ok(), "{:?}", speak_run.outcome);
    assert_eq!(std::fs::read(&pcm_path).expect("the samples"), [5, 0, 6, 0]);
}

// Negative: a refused voice is the user's to fix, and leaves no file behind.
#[tokio::test]
async fn should_report_a_refused_voice_as_a_usage_error_and_leave_no_file() {
    let speak_run = run_journey(
        "refused_voice",
        speech_stub_config(StubSpeechOutcome::Failed(SpeechFailure::invalid_request(
            "there is no voice \"am_michael\"; the voices are default",
        ))),
        speak_arguments(None),
    )
    .await;

    let Err(speak_error) = speak_run.outcome else {
        panic!("a refused voice must fail the journey");
    };
    assert!(speak_error.is_usage_error());
    assert!(speak_error.to_string().contains("there is no voice"));
    let leftover_audio_files = std::fs::read_dir(&speak_run.test_directory)
        .expect("the test directory is readable")
        .flatten()
        .filter(|directory_entry| {
            directory_entry
                .path()
                .extension()
                .is_some_and(|extension| extension == "wav")
        })
        .count();
    assert_eq!(leftover_audio_files, 0);
}
