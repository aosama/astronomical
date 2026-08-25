use std::{
    path::{Path, PathBuf},
    process::Stdio,
    time::Duration,
};

use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationCompletionReason, ChatGenerationSettings, ChatMessage,
    ChatToolChoice, ProtocolReader, ProtocolWriter, RequestId, WorkerCommand, WorkerEvent,
};
use astronomical_supervisor::{
    ResolvedRuntimeConfigResolver, WorkerProcess, WorkerTerminationOutcome,
};
use tokio::{process::Command, time::timeout};

#[tokio::test]
async fn should_serve_chat_through_child_process_stdio() {
    let worker_executable_path =
        std::env::var("CARGO_BIN_EXE_astronomical-inference-worker-scripted-worker")
            .expect("Cargo should provide the scripted worker path");
    let mut worker_process = WorkerProcess::launch(worker_executable_path)
        .await
        .expect("the scripted worker should launch");
    assert!(matches!(
        worker_process
            .next_event()
            .await
            .expect("the ready event should decode"),
        Some(WorkerEvent::Ready { .. })
    ));

    worker_process
        .start_generation(chat_command())
        .await
        .expect("the chat command should be written");
    assert!(matches!(
        worker_process
            .next_event()
            .await
            .expect("the completion event should decode"),
        Some(WorkerEvent::Completed {
            request_id,
            reason: ChatGenerationCompletionReason::EndOfSequence,
            ..
        }) if request_id == RequestId::new(41)
    ));
    worker_process
        .close()
        .await
        .expect("the worker should close");
}

#[tokio::test]
async fn should_force_terminate_and_reap_an_unresponsive_worker() {
    let worker_executable_path =
        std::env::var("CARGO_BIN_EXE_astronomical-inference-worker-stubborn-eof-worker")
            .expect("Cargo should provide the stubborn worker path");
    let mut worker_process = WorkerProcess::launch(worker_executable_path)
        .await
        .expect("the stubborn worker should launch");
    assert!(worker_process.next_event().await.is_ok());

    assert!(matches!(
        worker_process
            .force_terminate()
            .await
            .expect("forced termination should succeed"),
        WorkerTerminationOutcome::Forced { .. }
    ));
    assert!(worker_process.process_id().is_none());
}

#[tokio::test]
async fn should_start_the_production_worker_from_an_ipc_startup_configuration() {
    let temp_home = tempfile::tempdir().expect("temp home should be created");
    write_config(temp_home.path(), r#"{"prompt_cache_max_size_gb":50}"#);
    let worker_executable_path = std::env::var("CARGO_BIN_EXE_astronomical-inference-worker")
        .expect("Cargo should provide the production worker path");
    let worker_runtime_config = ResolvedRuntimeConfigResolver::for_development_home_directory(
        temp_home.path().to_path_buf(),
        PathBuf::from(&worker_executable_path),
    )
    .load()
    .expect("the supervisor-side worker configuration should resolve");
    let mut worker_process = Command::new(worker_executable_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("the model-free worker should start");
    let worker_stdout = worker_process
        .stdout
        .take()
        .expect("the worker should expose stdout");
    let worker_stdin = worker_process
        .stdin
        .take()
        .expect("the worker should expose stdin");
    let mut worker_command_writer = ProtocolWriter::new(worker_stdin);
    worker_command_writer
        .send_command(&WorkerCommand::InitializeWorker(
            worker_runtime_config.worker_startup_configuration(),
        ))
        .await
        .expect("the worker startup configuration should be written");
    let mut worker_event_reader = ProtocolReader::new(worker_stdout);
    let startup_event = worker_event_reader
        .next_event()
        .await
        .expect("the worker should emit a valid startup event");
    let Some(WorkerEvent::Idle {
        machine_mlx_memory_ceiling_bytes,
        effective_mlx_memory_ceiling_bytes,
        minimum_mlx_memory_ceiling_bytes,
    }) = startup_event
    else {
        panic!("the production worker should report an idle startup event");
    };
    assert!(machine_mlx_memory_ceiling_bytes > 0);
    assert!(effective_mlx_memory_ceiling_bytes > 0);
    assert!(minimum_mlx_memory_ceiling_bytes > 0);
    drop(worker_command_writer);
    let worker_exit_status = timeout(Duration::from_secs(3), worker_process.wait())
        .await
        .expect("the idle worker should exit after stdin closes")
        .expect("the idle worker should run");
    assert!(worker_exit_status.success());
}

#[tokio::test]
async fn should_exit_after_the_worker_future_finishes_while_stdin_remains_blocked() {
    let worker_executable_path =
        std::env::var("CARGO_BIN_EXE_astronomical-inference-worker-stubborn-eof-worker")
            .expect("Cargo should provide the worker fixture path");
    let mut worker_process = Command::new(worker_executable_path)
        .arg("--finish-with-blocked-stdin")
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("the blocked-stdin worker should start");
    let _open_worker_stdin = worker_process
        .stdin
        .take()
        .expect("the parent should keep worker stdin open");

    // The child only waits 100 milliseconds before its future completes, but parallel hermetic
    // execution can delay process scheduling. Keep the test bounded while allowing the operating
    // system enough time to observe the runtime's deliberate shutdown rather than failing on a
    // one-second scheduling race.
    let worker_exit_status = match timeout(Duration::from_secs(3), worker_process.wait()).await {
        Ok(wait_outcome) => wait_outcome.expect("the blocked-stdin worker should run"),
        Err(_) => {
            worker_process
                .kill()
                .await
                .expect("the timed-out blocked-stdin worker should be terminated");
            panic!("the worker process remained alive after its main future finished");
        }
    };
    assert!(worker_exit_status.success());
}

fn chat_command() -> ChatGenerationCommand {
    ChatGenerationCommand {
        request_id: RequestId::new(41),
        model: "astronomical/scripted-worker".to_owned(),
        messages: vec![ChatMessage::User {
            content: "hello".to_owned(),
            images: Vec::new(),
        }],
        tools: Vec::new(),
        tool_choice: ChatToolChoice::None,
        settings: ChatGenerationSettings {
            max_output_tokens: 1,
            temperature_thousandths: None,
            top_p_thousandths: None,
            seed: None,
            thinking_budget: None,
        },
        qwen_thinking_channel_seed: None,
    }
}

fn write_config(home_directory: &Path, config_json: &str) {
    let config_directory = home_directory.join(".astronomical-dev");
    std::fs::create_dir_all(&config_directory).expect("config directory should be created");
    std::fs::write(config_directory.join("config.json"), config_json)
        .expect("config file should be written");
}
