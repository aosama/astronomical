use super::scripted_model_factory_test_doubles::{
    FirstCreationFailsScriptedModelFactory, LazyScriptedModelFactory,
};
use super::support::{close_worker_transport, next_event, worker_model_configuration};
use crate::{
    EngineBackedWorker, MemoryCeilingUtilization, MlxActiveMemoryBreakdown, MlxMemoryTelemetry,
};
use astronomical_ipc_protocol::{
    ExpertMemoryMode, MAX_IPC_FRAME_BYTES, MlxMemorySnapshotSource, ProtocolReader, ProtocolWriter,
    WorkerCommand, WorkerEvent, WorkerExpertResidencySnapshot, WorkerRuntimeFeatureConfiguration,
};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::io::{duplex, split};
use tokio::time::timeout;
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
            scripted_replacement_required_reason: None,
        },
        0,
    );
    let (daemon_transport, worker_transport) = duplex(MAX_IPC_FRAME_BYTES * 2);
    let (daemon_reader_transport, daemon_writer_transport) = split(daemon_transport);
    let (worker_reader_transport, worker_writer_transport) = split(worker_transport);
    let mut daemon_reader = ProtocolReader::new(daemon_reader_transport);
    let mut daemon_writer = ProtocolWriter::new(daemon_writer_transport);
    let worker_task = tokio::spawn(async move {
        engine_worker
            .run(worker_reader_transport, worker_writer_transport)
            .await
    });

    assert_eq!(
        next_event(&mut daemon_reader).await,
        WorkerEvent::Idle {
            machine_mlx_memory_ceiling_bytes: 0,
            effective_mlx_memory_ceiling_bytes: 0,
            minimum_mlx_memory_ceiling_bytes: 1,
            wired_residency_budget_bytes: None,
        }
    );
    assert_eq!(model_factory_call_count.load(Ordering::SeqCst), 0);

    daemon_writer
        .send_command(&WorkerCommand::SwapModel {
            model_directory: "/models/requested-model".to_owned(),
            model_configuration: worker_model_configuration("requested-model"),
        })
        .await
        .expect("the worker should receive the first requested model");

    assert!(matches!(
        next_event(&mut daemon_reader).await,
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

    close_worker_transport(daemon_writer, worker_task).await;
}

#[tokio::test]
async fn should_apply_a_live_ceiling_within_the_machine_bound_to_the_next_model() {
    let model_factory_call_count = Arc::new(AtomicUsize::new(0));
    let model_creation_memory_limits = Arc::new(Mutex::new(Vec::new()));
    let engine_worker = EngineBackedWorker::idle_with_model_factory_and_machine_mlx_memory_ceiling(
        LazyScriptedModelFactory {
            model_factory_call_count,
            mlx_memory_limits: (32_000_000_000, 32_000_000_000),
            model_creation_memory_limits: Arc::clone(&model_creation_memory_limits),
            model_configurations: Arc::new(Mutex::new(Vec::new())),
            expert_memory_mode: Some(astronomical_ipc_protocol::ExpertMemoryMode::Paged),
            created_engine_idle_mlx_memory_telemetry: None,
            scripted_replacement_required_reason: None,
        },
        40_000_000_000,
        32_000_000_000,
    )
    .with_worker_runtime_feature_configuration(WorkerRuntimeFeatureConfiguration {
        configuration_generation: "generation-before-memory-update".to_owned(),
        persistent_prompt_cache_enabled: false,
        prompt_cache_maximum_size_bytes: 0,
        loaded_model: None,
    });
    let (daemon_transport, worker_transport) = duplex(MAX_IPC_FRAME_BYTES * 2);
    let (daemon_reader_transport, daemon_writer_transport) = split(daemon_transport);
    let (worker_reader_transport, worker_writer_transport) = split(worker_transport);
    let mut daemon_reader = ProtocolReader::new(daemon_reader_transport);
    let mut daemon_writer = ProtocolWriter::new(daemon_writer_transport);
    let worker_task = tokio::spawn(async move {
        engine_worker
            .run(worker_reader_transport, worker_writer_transport)
            .await
    });
    let _idle_event = next_event(&mut daemon_reader).await;
    let _idle_runtime_configuration = next_event(&mut daemon_reader).await;
    daemon_writer
        .send_command(&WorkerCommand::SwapModel {
            model_directory: "/models/first-model".to_owned(),
            model_configuration: worker_model_configuration("first-model"),
        })
        .await
        .expect("the first model should load");
    let _model_swapped_event = next_event(&mut daemon_reader).await;
    let _first_model_runtime_configuration = next_event(&mut daemon_reader).await;

    daemon_writer
        .send_command(&WorkerCommand::UpdateMlxMemoryLimit {
            effective_mlx_memory_ceiling_bytes: 40_000_000_000,
            configuration_generation: "generation-after-memory-update".to_owned(),
        })
        .await
        .expect("the live memory limit should reach the worker");
    assert!(matches!(
        next_event(&mut daemon_reader).await,
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

    daemon_writer
        .send_command(&WorkerCommand::SwapModel {
            model_directory: "/models/second-model".to_owned(),
            model_configuration: worker_model_configuration("second-model"),
        })
        .await
        .expect("the replacement model should receive the updated MLX limit pair");
    assert!(matches!(
        next_event(&mut daemon_reader).await,
        WorkerEvent::ModelSwapped { .. }
    ));
    assert!(matches!(
        next_event(&mut daemon_reader).await,
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
            (32_000_000_000, 32_000_000_000),
            (40_000_000_000, 32_000_000_000),
        ]
    );
    close_worker_transport(daemon_writer, worker_task).await;
}

#[tokio::test]
async fn should_apply_live_memory_updates_to_the_idle_worker_up_to_the_machine_bound() {
    let model_factory_call_count = Arc::new(AtomicUsize::new(0));
    let model_creation_memory_limits = Arc::new(Mutex::new(Vec::new()));
    let engine_worker = EngineBackedWorker::idle_with_model_factory_and_machine_mlx_memory_ceiling(
        LazyScriptedModelFactory {
            model_factory_call_count: Arc::clone(&model_factory_call_count),
            mlx_memory_limits: (35_000_000_000, 35_000_000_000),
            model_creation_memory_limits: Arc::clone(&model_creation_memory_limits),
            model_configurations: Arc::new(Mutex::new(Vec::new())),
            expert_memory_mode: Some(astronomical_ipc_protocol::ExpertMemoryMode::Resident),
            created_engine_idle_mlx_memory_telemetry: None,
            scripted_replacement_required_reason: None,
        },
        40_000_000_000,
        35_000_000_000,
    )
    .with_worker_runtime_feature_configuration(WorkerRuntimeFeatureConfiguration {
        configuration_generation: "generation-before-live-clamp".to_owned(),
        persistent_prompt_cache_enabled: false,
        prompt_cache_maximum_size_bytes: 0,
        loaded_model: None,
    });
    let (daemon_transport, worker_transport) = duplex(MAX_IPC_FRAME_BYTES * 2);
    let (daemon_reader_transport, daemon_writer_transport) = split(daemon_transport);
    let (worker_reader_transport, worker_writer_transport) = split(worker_transport);
    let mut daemon_reader = ProtocolReader::new(daemon_reader_transport);
    let mut daemon_writer = ProtocolWriter::new(daemon_writer_transport);
    let worker_task = tokio::spawn(async move {
        engine_worker
            .run(worker_reader_transport, worker_writer_transport)
            .await
    });

    let _idle_event = next_event(&mut daemon_reader).await;
    let _idle_runtime_configuration = next_event(&mut daemon_reader).await;
    daemon_writer
        .send_command(&WorkerCommand::UpdateMlxMemoryLimit {
            effective_mlx_memory_ceiling_bytes: 40_000_000_000,
            configuration_generation: "generation-after-live-clamp".to_owned(),
        })
        .await
        .expect("the live memory limit should reach the idle worker");
    assert!(matches!(
        next_event(&mut daemon_reader).await,
        WorkerEvent::MlxMemoryLimitChanged {
            effective_mlx_memory_ceiling_bytes: 40_000_000_000,
            ..
        }
    ));

    daemon_writer
        .send_command(&WorkerCommand::SwapModel {
            model_directory: "/models/clamped-model".to_owned(),
            model_configuration: worker_model_configuration("clamped-model"),
        })
        .await
        .expect("the model should load after the clamped setting");
    assert!(matches!(
        next_event(&mut daemon_reader).await,
        WorkerEvent::ModelSwapped { .. }
    ));
    assert!(matches!(
        next_event(&mut daemon_reader).await,
        WorkerEvent::RuntimeFeatureConfigurationApplied {
            worker_runtime_feature_configuration,
        } if worker_runtime_feature_configuration.configuration_generation
            == "generation-after-live-clamp"
    ));
    assert_eq!(model_factory_call_count.load(Ordering::SeqCst), 1);
    assert_eq!(
        *model_creation_memory_limits
            .lock()
            .unwrap_or_else(|poisoned_lock| poisoned_lock.into_inner()),
        vec![(40_000_000_000, 40_000_000_000)]
    );
    close_worker_transport(daemon_writer, worker_task).await;
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
    let (daemon_transport, worker_transport) = duplex(MAX_IPC_FRAME_BYTES * 2);
    let (daemon_reader_transport, daemon_writer_transport) = split(daemon_transport);
    let (worker_reader_transport, worker_writer_transport) = split(worker_transport);
    let mut daemon_reader = ProtocolReader::new(daemon_reader_transport);
    let mut daemon_writer = ProtocolWriter::new(daemon_writer_transport);
    let worker_task = tokio::spawn(async move {
        engine_worker
            .run(worker_reader_transport, worker_writer_transport)
            .await
    });

    assert_eq!(
        next_event(&mut daemon_reader).await,
        WorkerEvent::Idle {
            machine_mlx_memory_ceiling_bytes: 0,
            effective_mlx_memory_ceiling_bytes: 0,
            minimum_mlx_memory_ceiling_bytes: 1,
            wired_residency_budget_bytes: None,
        }
    );
    daemon_writer
        .send_command(&WorkerCommand::SwapModel {
            model_directory: "/models/invalid-model".to_owned(),
            model_configuration: worker_model_configuration("invalid-model"),
        })
        .await
        .expect("the worker should receive the invalid model selection");
    let model_swap_failure_event = timeout(Duration::from_millis(100), daemon_reader.next_event())
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

    daemon_writer
        .send_command(&WorkerCommand::SwapModel {
            model_directory: "/models/valid-model".to_owned(),
            model_configuration: worker_model_configuration("valid-model"),
        })
        .await
        .expect("the idle worker should accept a later valid model");
    assert!(matches!(
        next_event(&mut daemon_reader).await,
        WorkerEvent::ModelSwapped { .. }
    ));
    assert_eq!(model_factory_call_count.load(Ordering::SeqCst), 2);

    close_worker_transport(daemon_writer, worker_task).await;
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
            scripted_replacement_required_reason: None,
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
    let (daemon_transport, worker_transport) = duplex(MAX_IPC_FRAME_BYTES * 2);
    let (daemon_reader_transport, daemon_writer_transport) = split(daemon_transport);
    let mut daemon_writer = ProtocolWriter::new(daemon_writer_transport);
    let (worker_reader_transport, worker_writer_transport) = split(worker_transport);
    let mut daemon_reader = ProtocolReader::new(daemon_reader_transport);
    let worker_task = tokio::spawn(async move {
        engine_worker
            .run(worker_reader_transport, worker_writer_transport)
            .await
    });

    let _idle_event = next_event(&mut daemon_reader).await;
    let _idle_runtime_configuration = next_event(&mut daemon_reader).await;
    daemon_writer
        .send_command(&WorkerCommand::SwapModel {
            model_directory: "/models/observed-model".to_owned(),
            model_configuration: worker_model_configuration("observed-model"),
        })
        .await
        .expect("the observed model should load");
    let _model_swapped_event = next_event(&mut daemon_reader).await;
    let _model_runtime_configuration = next_event(&mut daemon_reader).await;

    let WorkerEvent::MlxMemorySample {
        mlx_memory_snapshot: Some(model_loaded_snapshot),
        ..
    } = next_event(&mut daemon_reader).await
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
    daemon_writer
        .send_command(&WorkerCommand::UpdateMlxMemoryLimit {
            effective_mlx_memory_ceiling_bytes: 24_000_000_000,
            configuration_generation: "limit-adjusted-generation".to_owned(),
        })
        .await
        .expect("the live memory limit should reach the worker");
    let WorkerEvent::MlxMemoryLimitChanged {
        mlx_memory_snapshot: Some(limit_adjusted_snapshot),
        ..
    } = next_event(&mut daemon_reader).await
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

    close_worker_transport(daemon_writer, worker_task).await;
}

#[tokio::test]
async fn should_report_restart_required_without_building_a_replacement_runtime() {
    let model_factory_call_count = Arc::new(AtomicUsize::new(0));
    let model_configurations = Arc::new(Mutex::new(Vec::new()));
    let engine_worker = EngineBackedWorker::idle_with_model_factory_and_machine_mlx_memory_ceiling(
        LazyScriptedModelFactory {
            model_factory_call_count: Arc::clone(&model_factory_call_count),
            mlx_memory_limits: (36_000_000_000, 36_000_000_000),
            model_creation_memory_limits: Arc::new(Mutex::new(Vec::new())),
            model_configurations: Arc::clone(&model_configurations),
            expert_memory_mode: Some(ExpertMemoryMode::Resident),
            created_engine_idle_mlx_memory_telemetry: None,
            scripted_replacement_required_reason: Some(
                "the resident engine cannot shrink below its loaded memory".to_owned(),
            ),
        },
        40_000_000_000,
        36_000_000_000,
    )
    .with_worker_runtime_feature_configuration(WorkerRuntimeFeatureConfiguration {
        configuration_generation: "generation-before-demotion".to_owned(),
        persistent_prompt_cache_enabled: false,
        prompt_cache_maximum_size_bytes: 0,
        loaded_model: None,
    });
    let (daemon_transport, worker_transport) = duplex(MAX_IPC_FRAME_BYTES * 2);
    let (daemon_reader_transport, daemon_writer_transport) = split(daemon_transport);
    let (worker_reader_transport, worker_writer_transport) = split(worker_transport);
    let mut daemon_reader = ProtocolReader::new(daemon_reader_transport);
    let mut daemon_writer = ProtocolWriter::new(daemon_writer_transport);
    let worker_task = tokio::spawn(async move {
        engine_worker
            .run(worker_reader_transport, worker_writer_transport)
            .await
    });

    let _idle_event = next_event(&mut daemon_reader).await;
    let _idle_runtime_configuration = next_event(&mut daemon_reader).await;
    daemon_writer
        .send_command(&WorkerCommand::SwapModel {
            model_directory: "/models/resident-model".to_owned(),
            model_configuration: worker_model_configuration("resident-model"),
        })
        .await
        .expect("the resident model should load");
    let _model_swapped_event = next_event(&mut daemon_reader).await;
    let _model_runtime_configuration = next_event(&mut daemon_reader).await;
    assert_eq!(model_factory_call_count.load(Ordering::SeqCst), 1);

    daemon_writer
        .send_command(&WorkerCommand::UpdateMlxMemoryLimit {
            effective_mlx_memory_ceiling_bytes: 19_000_000_000,
            configuration_generation: "generation-demotion-requested".to_owned(),
        })
        .await
        .expect("the demotion ceiling should reach the worker");

    // The only correct worker answer is the typed restart-required event: no
    // replacement runtime creation, no model swap, no limit adoption, and the
    // configuration generation stays untouched for the surviving worker.
    assert!(matches!(
        next_event(&mut daemon_reader).await,
        WorkerEvent::MlxMemoryLimitRestartRequired {
            requested_mlx_memory_ceiling_bytes: 19_000_000_000,
            effective_mlx_memory_ceiling_bytes: 19_000_000_000,
            configuration_generation,
            ..
        } if configuration_generation == "generation-demotion-requested"
    ));
    assert_eq!(
        model_factory_call_count.load(Ordering::SeqCst),
        1,
        "the demotion event must not construct a streaming replacement runtime"
    );
    assert_eq!(
        model_configurations
            .lock()
            .unwrap_or_else(|poisoned_lock| poisoned_lock.into_inner())
            .as_slice(),
        &[worker_model_configuration("resident-model")],
        "no additional model configuration may reach the factory during demotion"
    );

    daemon_writer
        .send_command(&WorkerCommand::UpdateMlxMemoryLimit {
            effective_mlx_memory_ceiling_bytes: 38_000_000_000,
            configuration_generation: "generation-after-demotion".to_owned(),
        })
        .await
        .expect("the surviving worker should still accept ordinary limits");
    assert!(matches!(
        next_event(&mut daemon_reader).await,
        WorkerEvent::MlxMemoryLimitChanged {
            effective_mlx_memory_ceiling_bytes: 38_000_000_000,
            ..
        }
    ));

    close_worker_transport(daemon_writer, worker_task).await;
}
