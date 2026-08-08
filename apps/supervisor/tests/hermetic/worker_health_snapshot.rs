use std::time::Duration;

use astronomical_ipc_protocol::{ChatModelCapabilities, MtpRuntimeState, WorkerPromptWorkReuse};
use astronomical_supervisor::{ChatGenerationExecutor, ServingSessionSnapshot, WorkerHealthStatus};
use tokio::time::{Instant, sleep, timeout};

use crate::common::supervisor::launch_test_executor;

#[tokio::test]
async fn should_publish_the_ready_model_identity_from_the_worker_readiness_event() {
    let worker_executable_path = std::env::var("CARGO_BIN_EXE_astronomical-supervisor-test-worker")
        .expect("Cargo should provide the scripted worker fixture path");
    let worker_executor = launch_test_executor(worker_executable_path)
        .await
        .expect("the worker-backed executor should launch the fixture worker");
    let health_deadline = Instant::now() + Duration::from_secs(2);

    loop {
        let worker_health_snapshot = worker_executor.worker_health_snapshot();
        if worker_health_snapshot.status == WorkerHealthStatus::Ready {
            assert_eq!(
                worker_health_snapshot.ready_model_id.as_deref(),
                Some("astronomical/test-worker")
            );
            assert_eq!(
                worker_health_snapshot.ready_model_capabilities,
                Some(ChatModelCapabilities {
                    supports_reasoning: true,
                    supports_tool_calls: true,
                    has_vision: true,
                    max_input_tokens: 241_664,
                    max_output_tokens: 20_480,
                    context_window: 262_144,
                })
            );
            break;
        }

        assert!(
            Instant::now() < health_deadline,
            "worker health did not become ready; last status was {:?}",
            worker_health_snapshot.status
        );
        timeout(Duration::from_millis(25), sleep(Duration::from_millis(25)))
            .await
            .expect("the health polling sleep should complete before its timeout");
    }

    worker_executor
        .shutdown()
        .await
        .expect("the worker-backed executor should shut down after the health test");
}

#[test]
fn should_clear_ready_model_identity_when_worker_health_is_unavailable() {
    let unavailable_health_snapshot =
        astronomical_supervisor::WorkerHealthSnapshot::unavailable(WorkerHealthStatus::Unavailable);

    assert_eq!(unavailable_health_snapshot.ready_model_id, None);
    assert_eq!(unavailable_health_snapshot.ready_model_capabilities, None);
}

#[test]
fn should_keep_worker_health_status_surface_minimal() {
    fn status_wire_text(worker_health_status: WorkerHealthStatus) -> &'static str {
        match worker_health_status {
            WorkerHealthStatus::Loading => "loading",
            WorkerHealthStatus::Ready => "ready",
            WorkerHealthStatus::Unavailable => "unavailable",
        }
    }

    assert_eq!(status_wire_text(WorkerHealthStatus::Loading), "loading");
    assert_eq!(status_wire_text(WorkerHealthStatus::Ready), "ready");
    assert_eq!(
        status_wire_text(WorkerHealthStatus::Unavailable),
        "unavailable"
    );
}

#[test]
fn should_average_only_requests_that_report_each_throughput_measurement() {
    let mut serving_session_snapshot = ServingSessionSnapshot::empty();

    serving_session_snapshot.record_completed_request(1_000, 100, Some(10.0), None);
    serving_session_snapshot.record_completed_request(2_000, 200, None, Some(20.0));
    serving_session_snapshot.record_prompt_work_reuse(WorkerPromptWorkReuse {
        target_eligible_token_count: 10_000,
        target_restored_token_count: 8_000,
        drafter_eligible_token_count: 50_000,
        drafter_restored_token_count: 40_000,
    });
    serving_session_snapshot.record_completed_request(3_000, 300, Some(30.0), Some(40.0));

    assert_eq!(serving_session_snapshot.completed_request_count, 3);
    assert_eq!(serving_session_snapshot.total_prompt_token_count, 6_000);
    assert_eq!(
        serving_session_snapshot.total_reused_prompt_token_count,
        600
    );
    assert_eq!(
        serving_session_snapshot.average_prefill_tok_per_second,
        20.0
    );
    assert_eq!(
        serving_session_snapshot.average_generation_tok_per_second,
        30.0
    );
    assert_eq!(
        serving_session_snapshot.target_prompt_work_token_count,
        10_000
    );
    assert_eq!(
        serving_session_snapshot.target_reused_prompt_work_token_count,
        8_000
    );
    assert_eq!(
        serving_session_snapshot.drafter_prompt_work_token_count,
        50_000
    );
    assert_eq!(
        serving_session_snapshot.drafter_reused_prompt_work_token_count,
        40_000
    );
}

#[test]
fn should_preserve_the_serving_session_when_the_resident_model_is_replaced() {
    let capabilities = ChatModelCapabilities {
        supports_reasoning: true,
        supports_tool_calls: true,
        has_vision: false,
        max_input_tokens: 100,
        max_output_tokens: 20,
        context_window: 120,
    };
    let mut previous_health_snapshot =
        astronomical_supervisor::WorkerHealthSnapshot::ready_with_model(
            "first-model".to_owned(),
            capabilities.clone(),
            MtpRuntimeState::Disabled,
            None,
        );
    previous_health_snapshot.mlx_memory_ceiling_bytes = 40_000;
    previous_health_snapshot
        .serving_session
        .record_completed_request(1_000, 750, Some(10.0), Some(20.0));

    let replacement_health_snapshot =
        astronomical_supervisor::WorkerHealthSnapshot::ready_with_replacement_model(
            "second-model".to_owned(),
            capabilities,
            3_000,
            MtpRuntimeState::Disabled,
            None,
            &previous_health_snapshot,
        );

    assert_eq!(replacement_health_snapshot.mlx_memory_ceiling_bytes, 40_000);
    assert_eq!(
        replacement_health_snapshot.minimum_mlx_memory_ceiling_bytes,
        3_000
    );
    assert_eq!(
        replacement_health_snapshot.serving_session,
        previous_health_snapshot.serving_session
    );
}
