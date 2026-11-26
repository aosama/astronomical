use super::ipc_worker_transport::open_ipc_worker_transport;
use super::scripted_model_factory_test_doubles::{
    FirstCreationFailsScriptedModelFactory, LazyScriptedModelFactory,
};
use super::support::{
    close_worker_transport, next_event, next_raw_event, worker_model_configuration,
};
use crate::{
    EngineBackedWorker, MemoryCeilingUtilization, MlxActiveMemoryBreakdown, MlxMemoryTelemetry,
};
use ipc_protocol::{
    ExpertMemoryMode, MlxMemorySnapshotSource, WorkerCommand, WorkerEvent,
    WorkerExpertResidencySnapshot, WorkerRuntimeFeatureConfiguration,
};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
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
            expert_memory_mode: Some(ipc_protocol::ExpertMemoryMode::Resident),
            created_engine_idle_mlx_memory_telemetry: None,
        },
        0,
    );
    let (mut daemon_reader, mut daemon_writer, worker_reader_transport, worker_writer_transport) =
        open_ipc_worker_transport();
    let worker_task =
        tokio::spawn(engine_worker.run(worker_reader_transport, worker_writer_transport));

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
            expert_memory_mode: Some(ipc_protocol::ExpertMemoryMode::Resident),
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
            expert_memory_mode: Some(ipc_protocol::ExpertMemoryMode::Paged),
            created_engine_idle_mlx_memory_telemetry: None,
        },
        40_000_000_000,
        32_000_000_000,
    )
    .with_worker_runtime_feature_configuration(WorkerRuntimeFeatureConfiguration {
        configuration_generation: "generation-before-memory-update".to_owned(),
        astro_ssd_prompt_cache_enabled: false,
        astro_ssd_prompt_cache_maximum_size_bytes: 0,
        loaded_model: None,
    });
    let (mut daemon_reader, mut daemon_writer, worker_reader_transport, worker_writer_transport) =
        open_ipc_worker_transport();
    let worker_task =
        tokio::spawn(engine_worker.run(worker_reader_transport, worker_writer_transport));
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
    assert_live_limit_event_seats_every_expert(next_event(&mut daemon_reader).await);

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
    assert_second_model_was_created_with_the_updated_limit_pair(&model_creation_memory_limits);
    close_worker_transport(daemon_writer, worker_task).await;
}

fn assert_live_limit_event_seats_every_expert(live_limit_event: WorkerEvent) {
    assert!(matches!(
        live_limit_event,
        WorkerEvent::MlxMemoryLimitChanged {
            effective_mlx_memory_ceiling_bytes: 40_000_000_000,
            expert_residency: Some(WorkerExpertResidencySnapshot {
                dense_layers: None,
                total_layer_count: 2,
                resident_expert_count: 2,
                resident_expert_payload_bytes: 8_000,
                complete_expert_payload_byte_count: 8_000,
                expert_capacity: 2,
                ..
            }),
            ..
        }
    ));
}

fn assert_second_model_was_created_with_the_updated_limit_pair(
    model_creation_memory_limits: &Mutex<Vec<(u64, u64)>>,
) {
    assert_eq!(
        *model_creation_memory_limits
            .lock()
            .unwrap_or_else(|poisoned_lock| poisoned_lock.into_inner()),
        vec![
            (32_000_000_000, 32_000_000_000),
            (40_000_000_000, 32_000_000_000),
        ]
    );
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
            expert_memory_mode: Some(ipc_protocol::ExpertMemoryMode::Resident),
            created_engine_idle_mlx_memory_telemetry: None,
        },
        40_000_000_000,
        35_000_000_000,
    )
    .with_worker_runtime_feature_configuration(WorkerRuntimeFeatureConfiguration {
        configuration_generation: "generation-before-live-clamp".to_owned(),
        astro_ssd_prompt_cache_enabled: false,
        astro_ssd_prompt_cache_maximum_size_bytes: 0,
        loaded_model: None,
    });
    let (mut daemon_reader, mut daemon_writer, worker_reader_transport, worker_writer_transport) =
        open_ipc_worker_transport();
    let worker_task =
        tokio::spawn(engine_worker.run(worker_reader_transport, worker_writer_transport));

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
    let (mut daemon_reader, mut daemon_writer, worker_reader_transport, worker_writer_transport) =
        open_ipc_worker_transport();
    let worker_task =
        tokio::spawn(engine_worker.run(worker_reader_transport, worker_writer_transport));

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
    let created_engine_idle_mlx_memory_telemetry = idle_telemetry_with_unused_ceiling_split();
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
        astro_ssd_prompt_cache_enabled: false,
        astro_ssd_prompt_cache_maximum_size_bytes: 0,
        loaded_model: None,
    });
    let (mut daemon_reader, mut daemon_writer, worker_reader_transport, worker_writer_transport) =
        open_ipc_worker_transport();
    let worker_task =
        tokio::spawn(engine_worker.run(worker_reader_transport, worker_writer_transport));

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

    assert_model_loaded_sample_keeps_the_unused_ceiling_split(next_event(&mut daemon_reader).await);

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
    assert_limit_adjusted_sample_keeps_the_unused_ceiling_split(
        next_event(&mut daemon_reader).await,
    );

    close_worker_transport(daemon_writer, worker_task).await;
}

fn idle_telemetry_with_unused_ceiling_split() -> MlxMemoryTelemetry {
    let utilization_split = MemoryCeilingUtilization {
        mlx_active_memory_ceiling_bytes: 23_000_000_000,
        active_memory_bytes: 16_000_000_000,
        unused_headroom_bytes: 7_000_000_000,
        reserved_model_core_slack_bytes: 0,
        reserved_context_growth_bytes: 2_000_000_000,
        reserved_activation_and_workspace_bytes: 1_000_000_000,
        unseated_expert_entitlement_bytes: 4_000_000_000,
        unneeded_headroom_bytes: 0,
        unexplained_headroom_bytes: 0,
        owner_overrun_bytes: 0,
    };
    MlxMemoryTelemetry::new(
        16_000_000_000,
        0,
        16_000_000_000,
        MlxActiveMemoryBreakdown {
            expert_payload_bytes: 12_000_000_000,
            model_core_payload_bytes: 3_000_000_000,
            context_state_payload_bytes: 0,
        },
    )
    .with_memory_ceiling_utilization(utilization_split)
}

fn assert_model_loaded_sample_keeps_the_unused_ceiling_split(model_loaded_event: WorkerEvent) {
    let WorkerEvent::MlxMemorySample {
        mlx_memory_snapshot: Some(model_loaded_snapshot),
        ..
    } = model_loaded_event
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
}

fn assert_limit_adjusted_sample_keeps_the_unused_ceiling_split(limit_adjusted_event: WorkerEvent) {
    let WorkerEvent::MlxMemoryLimitChanged {
        mlx_memory_snapshot: Some(limit_adjusted_snapshot),
        ..
    } = limit_adjusted_event
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
}

/// The daemon's status relies on the opening report arriving before the swap's
/// acknowledgement, so the report can never outlive the load it describes.
#[tokio::test]
async fn should_open_a_requested_swap_by_naming_the_model() {
    let engine_worker = EngineBackedWorker::idle_with_model_factory(
        LazyScriptedModelFactory {
            model_factory_call_count: Arc::new(AtomicUsize::new(0)),
            mlx_memory_limits: (0, 0),
            model_creation_memory_limits: Arc::new(Mutex::new(Vec::new())),
            model_configurations: Arc::new(Mutex::new(Vec::new())),
            expert_memory_mode: Some(ExpertMemoryMode::Resident),
            created_engine_idle_mlx_memory_telemetry: None,
        },
        0,
    );
    let (mut daemon_reader, mut daemon_writer, worker_reader_transport, worker_writer_transport) =
        open_ipc_worker_transport();
    let worker_task =
        tokio::spawn(engine_worker.run(worker_reader_transport, worker_writer_transport));
    assert!(matches!(
        next_raw_event(&mut daemon_reader).await,
        WorkerEvent::Idle { .. }
    ));

    daemon_writer
        .send_command(&WorkerCommand::SwapModel {
            model_directory: "/models/requested-model".to_owned(),
            model_configuration: worker_model_configuration("requested-model"),
        })
        .await
        .expect("the worker should receive the requested model");

    match next_raw_event(&mut daemon_reader).await {
        WorkerEvent::ModelLoadInFlight {
            model_load_in_flight,
        } => assert_eq!(
            model_load_in_flight.model_id.as_deref(),
            Some("requested-model")
        ),
        unexpected_event => panic!("the load should open with a report: {unexpected_event:?}"),
    }
    assert!(matches!(
        next_event(&mut daemon_reader).await,
        WorkerEvent::ModelSwapped { .. }
    ));

    close_worker_transport(daemon_writer, worker_task).await;
}
