//! Live MLX memory ceiling updates against the worker-lifetime wired budget.

use super::ipc_worker_transport::open_ipc_worker_transport;
use super::scripted_model_factory_test_doubles::{LazyScriptedModelFactory, ScriptedFactoryWorker};
use super::support::{close_worker_transport, next_event};
use ipc_protocol::{WorkerCommand, WorkerEvent};
use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, Mutex};

const MACHINE_MLX_MEMORY_CEILING_BYTES: u64 = 42_949_672_960;
const WIRED_RESIDENCY_BUDGET_BYTES: u64 = 42_097_180_672;
/// The wired budget minus its 2.5% Metal working-set margin.
const PLANNABLE_WIRED_RESIDENCY_BYTES: u64 = 41_044_751_172;
const STARTUP_MLX_MEMORY_CEILING_BYTES: u64 = 23_622_320_128;

async fn applied_live_mlx_memory_ceiling_bytes(
    wired_residency_budget_bytes: Option<u64>,
    requested_mlx_memory_ceiling_bytes: u64,
) -> u64 {
    let engine_worker =
        ScriptedFactoryWorker::idle_with_model_factory_and_machine_mlx_memory_ceiling(
            LazyScriptedModelFactory {
                model_factory_call_count: Arc::new(AtomicUsize::new(0)),
                mlx_memory_limits: (
                    STARTUP_MLX_MEMORY_CEILING_BYTES,
                    STARTUP_MLX_MEMORY_CEILING_BYTES,
                ),
                model_creation_memory_limits: Arc::new(Mutex::new(Vec::new())),
                model_configurations: Arc::new(Mutex::new(Vec::new())),
                expert_memory_mode: Some(ipc_protocol::ExpertMemoryMode::Resident),
                created_engine_idle_mlx_memory_telemetry: None,
            },
            MACHINE_MLX_MEMORY_CEILING_BYTES,
            STARTUP_MLX_MEMORY_CEILING_BYTES,
        )
        .with_wired_residency_budget_bytes(wired_residency_budget_bytes);
    let (mut daemon_reader, mut daemon_writer, worker_reader_transport, worker_writer_transport) =
        open_ipc_worker_transport();
    let worker_task =
        tokio::spawn(engine_worker.run(worker_reader_transport, worker_writer_transport));

    let _idle_event = next_event(&mut daemon_reader).await;
    daemon_writer
        .send_command(&WorkerCommand::UpdateMlxMemoryLimit {
            effective_mlx_memory_ceiling_bytes: requested_mlx_memory_ceiling_bytes,
            configuration_generation: "generation-after-live-raise".to_owned(),
        })
        .await
        .expect("the live memory limit should reach the idle worker");
    let WorkerEvent::MlxMemoryLimitChanged {
        effective_mlx_memory_ceiling_bytes,
        ..
    } = next_event(&mut daemon_reader).await
    else {
        panic!("the idle worker should acknowledge the live memory limit");
    };
    close_worker_transport(daemon_writer, worker_task).await;
    effective_mlx_memory_ceiling_bytes
}

#[tokio::test]
async fn should_apply_a_live_raise_above_the_startup_ceiling_inside_the_wired_budget() {
    let requested_mlx_memory_ceiling_bytes = 32_212_254_720;

    assert_eq!(
        applied_live_mlx_memory_ceiling_bytes(
            Some(WIRED_RESIDENCY_BUDGET_BYTES),
            requested_mlx_memory_ceiling_bytes
        )
        .await,
        requested_mlx_memory_ceiling_bytes
    );
}

#[tokio::test]
async fn should_apply_a_live_raise_that_equals_the_plannable_wired_residency_exactly() {
    assert_eq!(
        applied_live_mlx_memory_ceiling_bytes(
            Some(WIRED_RESIDENCY_BUDGET_BYTES),
            PLANNABLE_WIRED_RESIDENCY_BYTES
        )
        .await,
        PLANNABLE_WIRED_RESIDENCY_BYTES
    );
}

#[tokio::test]
async fn should_clamp_a_live_raise_to_the_wired_budget_to_its_plannable_part() {
    assert_eq!(
        applied_live_mlx_memory_ceiling_bytes(
            Some(WIRED_RESIDENCY_BUDGET_BYTES),
            WIRED_RESIDENCY_BUDGET_BYTES
        )
        .await,
        PLANNABLE_WIRED_RESIDENCY_BYTES
    );
}

#[tokio::test]
async fn should_clamp_a_live_raise_above_the_wired_budget_to_the_plannable_wired_residency() {
    assert_eq!(
        applied_live_mlx_memory_ceiling_bytes(
            Some(WIRED_RESIDENCY_BUDGET_BYTES),
            MACHINE_MLX_MEMORY_CEILING_BYTES
        )
        .await,
        PLANNABLE_WIRED_RESIDENCY_BYTES
    );
}

#[tokio::test]
async fn should_keep_the_machine_bound_when_no_wired_budget_was_granted() {
    assert_eq!(
        applied_live_mlx_memory_ceiling_bytes(None, MACHINE_MLX_MEMORY_CEILING_BYTES).await,
        MACHINE_MLX_MEMORY_CEILING_BYTES
    );
}
