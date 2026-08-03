use super::*;

#[tokio::test]
async fn should_report_the_loaded_engines_astronomical_optimized_expert_storage_format() {
    let engine_worker = EngineBackedWorker::new(
        ScriptedChatProcessor::new(),
        ScriptedChatEngine::new()
            .with_expert_storage_format(ExpertStorageFormat::AstronomicalAligned),
    );
    let (supervisor_transport, worker_transport) = duplex(MAX_IPC_FRAME_BYTES * 2);
    let (supervisor_reader_transport, supervisor_writer_transport) = split(supervisor_transport);
    let (worker_reader_transport, worker_writer_transport) = split(worker_transport);
    let mut supervisor_reader = ProtocolReader::new(supervisor_reader_transport);
    let worker_task = tokio::spawn(async move {
        engine_worker
            .run(worker_reader_transport, worker_writer_transport)
            .await
    });

    assert_eq!(
        next_event(&mut supervisor_reader).await,
        ready_event_with_expert_storage_format(ExpertStorageFormat::AstronomicalAligned)
    );

    close_worker_transport(
        ProtocolWriter::new(supervisor_writer_transport),
        worker_task,
    )
    .await;
}

#[tokio::test]
async fn should_report_the_loaded_engines_target_only_mtp_runtime_state() {
    let engine_worker = EngineBackedWorker::new(
        ScriptedChatProcessor::new(),
        ScriptedChatEngine::new().with_mtp_runtime_state(MtpRuntimeState::TargetOnly, None),
    );
    let (supervisor_transport, worker_transport) = duplex(MAX_IPC_FRAME_BYTES * 2);
    let (supervisor_reader_transport, supervisor_writer_transport) = split(supervisor_transport);
    let (worker_reader_transport, worker_writer_transport) = split(worker_transport);
    let mut supervisor_reader = ProtocolReader::new(supervisor_reader_transport);
    let worker_task = tokio::spawn(async move {
        engine_worker
            .run(worker_reader_transport, worker_writer_transport)
            .await
    });

    assert_eq!(
        next_event(&mut supervisor_reader).await,
        ready_event_with_load_details(
            ExpertStorageFormat::StandardSafetensors,
            MtpRuntimeState::TargetOnly,
            None,
        )
    );

    close_worker_transport(
        ProtocolWriter::new(supervisor_writer_transport),
        worker_task,
    )
    .await;
}

#[tokio::test]
async fn should_return_an_error_when_the_unit_model_factory_is_called() {
    let model_creation_outcome =
        <() as ModelFactory<ScriptedChatProcessor, ScriptedChatEngine>>::create(
            &(),
            "/unused/model",
            1,
        )
        .await;

    assert_eq!(
        model_creation_outcome.err().as_deref(),
        Some("model swapping is unavailable because no model factory was configured")
    );
}

#[tokio::test]
async fn should_wait_for_a_swap_command_before_loading_an_idle_worker_model() {
    let model_factory_call_count = Arc::new(AtomicUsize::new(0));
    let mlx_memory_ceiling_bytes = Arc::new(AtomicU64::new(0));
    let engine_worker = EngineBackedWorker::idle_with_model_factory(
        LazyScriptedModelFactory {
            model_factory_call_count: Arc::clone(&model_factory_call_count),
            mlx_memory_ceiling_bytes,
        },
        0,
    );
    let (supervisor_transport, worker_transport) = duplex(MAX_IPC_FRAME_BYTES * 2);
    let (supervisor_reader_transport, supervisor_writer_transport) = split(supervisor_transport);
    let (worker_reader_transport, worker_writer_transport) = split(worker_transport);
    let mut supervisor_reader = ProtocolReader::new(supervisor_reader_transport);
    let mut supervisor_writer = ProtocolWriter::new(supervisor_writer_transport);
    let worker_task = tokio::spawn(async move {
        engine_worker
            .run(worker_reader_transport, worker_writer_transport)
            .await
    });

    assert_eq!(
        next_event(&mut supervisor_reader).await,
        WorkerEvent::Idle {
            machine_mlx_memory_ceiling_bytes: 0,
            effective_mlx_memory_ceiling_bytes: 0,
            minimum_mlx_memory_ceiling_bytes: 1,
        }
    );
    assert_eq!(model_factory_call_count.load(Ordering::SeqCst), 0);

    supervisor_writer
        .send_command(&WorkerCommand::SwapModel {
            model_directory: "/models/requested-model".to_owned(),
            max_output_tokens: 20_480,
        })
        .await
        .expect("the worker should receive the first requested model");

    assert!(matches!(
        next_event(&mut supervisor_reader).await,
        WorkerEvent::ModelSwapped { .. }
    ));
    assert_eq!(model_factory_call_count.load(Ordering::SeqCst), 1);

    close_worker_transport(supervisor_writer, worker_task).await;
}

#[tokio::test]
async fn should_update_the_model_factory_memory_ceiling_after_a_loaded_engine_changes() {
    let model_factory_call_count = Arc::new(AtomicUsize::new(0));
    let mlx_memory_ceiling_bytes = Arc::new(AtomicU64::new(40_000_000_000));
    let engine_worker = EngineBackedWorker::idle_with_model_factory(
        LazyScriptedModelFactory {
            model_factory_call_count,
            mlx_memory_ceiling_bytes: Arc::clone(&mlx_memory_ceiling_bytes),
        },
        40_000_000_000,
    );
    let (supervisor_transport, worker_transport) = duplex(MAX_IPC_FRAME_BYTES * 2);
    let (supervisor_reader_transport, supervisor_writer_transport) = split(supervisor_transport);
    let (worker_reader_transport, worker_writer_transport) = split(worker_transport);
    let mut supervisor_reader = ProtocolReader::new(supervisor_reader_transport);
    let mut supervisor_writer = ProtocolWriter::new(supervisor_writer_transport);
    let worker_task = tokio::spawn(async move {
        engine_worker
            .run(worker_reader_transport, worker_writer_transport)
            .await
    });
    let _idle_event = next_event(&mut supervisor_reader).await;
    supervisor_writer
        .send_command(&WorkerCommand::SwapModel {
            model_directory: "/models/first-model".to_owned(),
            max_output_tokens: 20_480,
        })
        .await
        .expect("the first model should load");
    let _model_swapped_event = next_event(&mut supervisor_reader).await;

    supervisor_writer
        .send_command(&WorkerCommand::UpdateMlxMemoryLimit {
            effective_mlx_memory_ceiling_bytes: 8_000_000_000,
        })
        .await
        .expect("the live memory limit should reach the worker");
    assert!(matches!(
        next_event(&mut supervisor_reader).await,
        WorkerEvent::MlxMemoryLimitChanged {
            effective_mlx_memory_ceiling_bytes: 8_000_000_000,
            ..
        }
    ));

    assert_eq!(
        mlx_memory_ceiling_bytes.load(Ordering::SeqCst),
        8_000_000_000
    );
    close_worker_transport(supervisor_writer, worker_task).await;
}

#[tokio::test]
async fn should_remain_idle_after_the_first_model_creation_fails() {
    let model_factory_call_count = Arc::new(AtomicUsize::new(0));
    let engine_worker = EngineBackedWorker::idle_with_model_factory(
        FirstCreationFailsScriptedModelFactory {
            model_factory_call_count: Arc::clone(&model_factory_call_count),
        },
        0,
    );
    let (supervisor_transport, worker_transport) = duplex(MAX_IPC_FRAME_BYTES * 2);
    let (supervisor_reader_transport, supervisor_writer_transport) = split(supervisor_transport);
    let (worker_reader_transport, worker_writer_transport) = split(worker_transport);
    let mut supervisor_reader = ProtocolReader::new(supervisor_reader_transport);
    let mut supervisor_writer = ProtocolWriter::new(supervisor_writer_transport);
    let worker_task = tokio::spawn(async move {
        engine_worker
            .run(worker_reader_transport, worker_writer_transport)
            .await
    });

    assert_eq!(
        next_event(&mut supervisor_reader).await,
        WorkerEvent::Idle {
            machine_mlx_memory_ceiling_bytes: 0,
            effective_mlx_memory_ceiling_bytes: 0,
            minimum_mlx_memory_ceiling_bytes: 1,
        }
    );
    supervisor_writer
        .send_command(&WorkerCommand::SwapModel {
            model_directory: "/models/invalid-model".to_owned(),
            max_output_tokens: 20_480,
        })
        .await
        .expect("the worker should receive the invalid model selection");
    let model_swap_failure_event =
        timeout(Duration::from_millis(100), supervisor_reader.next_event())
            .await
            .expect("the worker should report model creation failure without waiting for a timeout")
            .expect("the model failure event should be valid")
            .expect("the worker transport should remain open");
    assert_eq!(
        model_swap_failure_event,
        WorkerEvent::ModelSwapFailed {
            loaded_model_remains_ready: false,
            model_load_failure_reason: "the scripted first model is invalid".to_owned(),
        }
    );

    supervisor_writer
        .send_command(&WorkerCommand::SwapModel {
            model_directory: "/models/valid-model".to_owned(),
            max_output_tokens: 20_480,
        })
        .await
        .expect("the idle worker should accept a later valid model");
    assert!(matches!(
        next_event(&mut supervisor_reader).await,
        WorkerEvent::ModelSwapped { .. }
    ));
    assert_eq!(model_factory_call_count.load(Ordering::SeqCst), 2);

    close_worker_transport(supervisor_writer, worker_task).await;
}
