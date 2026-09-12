use super::*;

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
        ready_event_with_load_details(MtpRuntimeState::TargetOnly, None,)
    );

    close_worker_transport(
        ProtocolWriter::new(supervisor_writer_transport),
        worker_task,
    )
    .await;
}

#[tokio::test]
async fn should_report_active_speculative_prefill_identity_when_the_draft_is_loaded() {
    let engine_worker = EngineBackedWorker::new(
        ScriptedChatProcessor::new(),
        ScriptedChatEngine::new().with_speculative_prefill_runtime(
            SpeculativePrefillRuntimeState::Active,
            None,
            Some("example/speculative-draft".to_owned()),
            Some("draft-revision-1".to_owned()),
        ),
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
        ready_event_with_speculative_prefill_load_details(
            MtpRuntimeState::Disabled,
            None,
            SpeculativePrefillRuntimeState::Active,
            None,
            Some("example/speculative-draft".to_owned()),
            Some("draft-revision-1".to_owned()),
        )
    );

    close_worker_transport(
        ProtocolWriter::new(supervisor_writer_transport),
        worker_task,
    )
    .await;
}

#[tokio::test]
async fn should_wait_for_a_swap_command_before_loading_an_idle_worker_model() {
    let model_factory_call_count = Arc::new(AtomicUsize::new(0));
    let model_configurations = Arc::new(Mutex::new(Vec::new()));
    let engine_worker = EngineBackedWorker::idle_with_model_factory(
        LazyScriptedModelFactory {
            model_factory_call_count: Arc::clone(&model_factory_call_count),
            mlx_memory_limits: (0, 0),
            model_creation_memory_limits: Arc::new(Mutex::new(Vec::new())),
            model_configurations: Arc::clone(&model_configurations),
            expert_memory_mode: Some(astronomical_ipc_protocol::ExpertMemoryMode::Resident),
            created_engine_idle_mlx_memory_telemetry: None,
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
            model_configuration: worker_model_configuration("requested-model"),
        })
        .await
        .expect("the worker should receive the first requested model");

    assert!(matches!(
        next_event(&mut supervisor_reader).await,
        WorkerEvent::ModelSwapped {
            expert_memory_mode: Some(astronomical_ipc_protocol::ExpertMemoryMode::Resident),
            ..
        }
    ));
    assert_eq!(model_factory_call_count.load(Ordering::SeqCst), 1);
    assert_eq!(
        model_configurations
            .lock()
            .unwrap_or_else(|poisoned_lock| poisoned_lock.into_inner())
            .as_slice(),
        &[worker_model_configuration("requested-model")]
    );

    close_worker_transport(supervisor_writer, worker_task).await;
}

#[tokio::test]
async fn should_reuse_the_live_memory_limit_and_configuration_generation_for_the_next_model() {
    let model_factory_call_count = Arc::new(AtomicUsize::new(0));
    let model_creation_memory_limits = Arc::new(Mutex::new(Vec::new()));
    let engine_worker = EngineBackedWorker::idle_with_model_factory_and_machine_mlx_memory_ceiling(
        LazyScriptedModelFactory {
            model_factory_call_count,
            mlx_memory_limits: (38_000_000_000, 38_000_000_000),
            model_creation_memory_limits: Arc::clone(&model_creation_memory_limits),
            model_configurations: Arc::new(Mutex::new(Vec::new())),
            expert_memory_mode: Some(astronomical_ipc_protocol::ExpertMemoryMode::Paged),
            created_engine_idle_mlx_memory_telemetry: None,
        },
        40_000_000_000,
        38_000_000_000,
    )
    .with_worker_runtime_feature_configuration(WorkerRuntimeFeatureConfiguration {
        configuration_generation: "generation-before-memory-update".to_owned(),
        persistent_prompt_cache_enabled: false,
        prompt_cache_maximum_size_bytes: 0,
        loaded_model: None,
    });
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
    let _idle_runtime_configuration = next_event(&mut supervisor_reader).await;
    supervisor_writer
        .send_command(&WorkerCommand::SwapModel {
            model_directory: "/models/first-model".to_owned(),
            model_configuration: worker_model_configuration("first-model"),
        })
        .await
        .expect("the first model should load");
    let _model_swapped_event = next_event(&mut supervisor_reader).await;
    let _first_model_runtime_configuration = next_event(&mut supervisor_reader).await;

    supervisor_writer
        .send_command(&WorkerCommand::UpdateMlxMemoryLimit {
            effective_mlx_memory_ceiling_bytes: 40_000_000_000,
            configuration_generation: "generation-after-memory-update".to_owned(),
        })
        .await
        .expect("the live memory limit should reach the worker");
    assert!(matches!(
        next_event(&mut supervisor_reader).await,
        WorkerEvent::MlxMemoryLimitChanged {
            effective_mlx_memory_ceiling_bytes: 40_000_000_000,
            expert_residency: Some(WorkerExpertResidencySnapshot {
                total_layer_count: 2,
                resident_expert_count: 2,
                resident_expert_payload_bytes: 8_000,
            }),
            ..
        }
    ));

    supervisor_writer
        .send_command(&WorkerCommand::SwapModel {
            model_directory: "/models/second-model".to_owned(),
            model_configuration: worker_model_configuration("second-model"),
        })
        .await
        .expect("the replacement model should receive the updated MLX limit pair");
    assert!(matches!(
        next_event(&mut supervisor_reader).await,
        WorkerEvent::ModelSwapped { .. }
    ));
    assert!(matches!(
        next_event(&mut supervisor_reader).await,
        WorkerEvent::RuntimeFeatureConfigurationApplied {
            worker_runtime_feature_configuration,
        } if worker_runtime_feature_configuration.configuration_generation
            == "generation-after-memory-update"
    ));
    assert_eq!(
        *model_creation_memory_limits
            .lock()
            .unwrap_or_else(|poisoned_lock| poisoned_lock.into_inner()),
        vec![
            (38_000_000_000, 38_000_000_000),
            (40_000_000_000, 38_000_000_000),
        ]
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
            model_configuration: worker_model_configuration("invalid-model"),
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
            model_configuration: worker_model_configuration("valid-model"),
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

#[tokio::test]
async fn should_publish_the_unused_ceiling_split_on_model_loaded_and_limit_adjusted_samples() {
    let model_factory_call_count = Arc::new(AtomicUsize::new(0));
    let utilization_split = MemoryCeilingUtilization {
        mlx_active_memory_ceiling_bytes: 23_000_000_000,
        active_memory_bytes: 16_000_000_000,
        unused_headroom_bytes: 7_000_000_000,
        reserved_model_core_slack_bytes: 0,
        reserved_context_growth_bytes: 2_000_000_000,
        reserved_activation_and_workspace_bytes: 1_000_000_000,
        unseated_expert_entitlement_bytes: 4_000_000_000,
        speculative_draft_payload_bytes: 0,
        unexplained_headroom_bytes: 0,
        owner_overrun_bytes: 0,
    };
    let created_engine_idle_mlx_memory_telemetry = MlxMemoryTelemetry::new(
        16_000_000_000,
        0,
        16_000_000_000,
        MlxActiveMemoryBreakdown {
            expert_payload_bytes: 12_000_000_000,
            model_core_payload_bytes: 3_000_000_000,
            context_state_payload_bytes: 0,
            speculative_prefill_draft_memory_bytes: 0,
        },
    )
    .with_memory_ceiling_utilization(utilization_split);
    let engine_worker = EngineBackedWorker::idle_with_model_factory_and_machine_mlx_memory_ceiling(
        LazyScriptedModelFactory {
            model_factory_call_count,
            mlx_memory_limits: (38_000_000_000, 38_000_000_000),
            model_creation_memory_limits: Arc::new(Mutex::new(Vec::new())),
            model_configurations: Arc::new(Mutex::new(Vec::new())),
            expert_memory_mode: Some(ExpertMemoryMode::Paged),
            created_engine_idle_mlx_memory_telemetry: Some(
                created_engine_idle_mlx_memory_telemetry,
            ),
        },
        40_000_000_000,
        38_000_000_000,
    )
    .with_worker_runtime_feature_configuration(WorkerRuntimeFeatureConfiguration {
        configuration_generation: "startup-generation".to_owned(),
        persistent_prompt_cache_enabled: false,
        prompt_cache_maximum_size_bytes: 0,
        loaded_model: None,
    });
    let (supervisor_transport, worker_transport) = duplex(MAX_IPC_FRAME_BYTES * 2);
    let (supervisor_reader_transport, supervisor_writer_transport) = split(supervisor_transport);
    let mut supervisor_writer = ProtocolWriter::new(supervisor_writer_transport);
    let (worker_reader_transport, worker_writer_transport) = split(worker_transport);
    let mut supervisor_reader = ProtocolReader::new(supervisor_reader_transport);
    let worker_task = tokio::spawn(async move {
        engine_worker
            .run(worker_reader_transport, worker_writer_transport)
            .await
    });

    let _idle_event = next_event(&mut supervisor_reader).await;
    let _idle_runtime_configuration = next_event(&mut supervisor_reader).await;
    supervisor_writer
        .send_command(&WorkerCommand::SwapModel {
            model_directory: "/models/observed-model".to_owned(),
            model_configuration: worker_model_configuration("observed-model"),
        })
        .await
        .expect("the observed model should load");
    let _model_swapped_event = next_event(&mut supervisor_reader).await;
    let _model_runtime_configuration = next_event(&mut supervisor_reader).await;

    let WorkerEvent::MlxMemorySample {
        mlx_memory_snapshot: Some(model_loaded_snapshot),
        ..
    } = next_event(&mut supervisor_reader).await
    else {
        panic!("load-complete must emit an MLX memory sample after the swap");
    };
    assert_eq!(
        model_loaded_snapshot.source,
        MlxMemorySnapshotSource::ModelLoaded
    );
    let model_loaded_utilization = model_loaded_snapshot
        .memory_ceiling_utilization
        .expect("the model-loaded sample must carry the unused-ceiling split the menu shows");
    assert_eq!(
        model_loaded_utilization.unused_headroom_bytes,
        7_000_000_000
    );
    assert_eq!(model_loaded_utilization.unexplained_headroom_bytes, 0);
    assert_eq!(model_loaded_utilization.owner_overrun_bytes, 0);

    // The live-limit event lands after the ModelLoaded sample, so a split-less
    // telemetry here would overwrite the load-complete split on the status the
    // menu paints (issue #510).
    supervisor_writer
        .send_command(&WorkerCommand::UpdateMlxMemoryLimit {
            effective_mlx_memory_ceiling_bytes: 24_000_000_000,
            configuration_generation: "limit-adjusted-generation".to_owned(),
        })
        .await
        .expect("the live memory limit should reach the worker");
    let WorkerEvent::MlxMemoryLimitChanged {
        mlx_memory_snapshot: Some(limit_adjusted_snapshot),
        ..
    } = next_event(&mut supervisor_reader).await
    else {
        panic!("the live limit must publish an MLX memory sample");
    };
    assert_eq!(
        limit_adjusted_snapshot.source,
        MlxMemorySnapshotSource::MemoryLimitAdjusted
    );
    let limit_adjusted_utilization = limit_adjusted_snapshot
        .memory_ceiling_utilization
        .expect("the limit-adjusted sample must keep the unused-ceiling split");
    assert_eq!(
        limit_adjusted_utilization.unused_headroom_bytes,
        7_000_000_000
    );
    assert_eq!(limit_adjusted_utilization.unexplained_headroom_bytes, 0);
    assert_eq!(limit_adjusted_utilization.owner_overrun_bytes, 0);

    close_worker_transport(supervisor_writer, worker_task).await;
}
