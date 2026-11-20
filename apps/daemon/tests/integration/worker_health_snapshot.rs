use std::time::Duration;

use astronomical_daemon::{
    ChatGenerationExecutor, CompletedRequestWork, PendingAstroSsdPromptCacheClear,
    ServingSessionSnapshot, WorkerHealthSnapshot, WorkerHealthStatus,
};
use ipc_protocol::{
    ChatModelCapabilities, ImageGenerationCapabilities, WorkerModelCapabilities,
    WorkerPromptWorkReuse,
};
use tokio::time::{Instant, sleep, timeout};

use crate::support::daemon;

#[tokio::test]
async fn should_publish_the_ready_model_identity_from_the_worker_readiness_event() {
    let worker_executable_path = std::env::var("CARGO_BIN_EXE_astronomical-daemon-test-worker")
        .expect("Cargo should provide the scripted worker fixture path");
    let worker_executor = daemon::launch_test_executor(worker_executable_path)
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
                Some(WorkerModelCapabilities::chat_and_image(
                    ChatModelCapabilities {
                        supports_reasoning: true,
                        supports_tool_calls: true,
                        has_vision: true,
                        max_input_tokens: 241_664,
                        max_output_tokens: 20_480,
                        context_window: 262_144,
                    },
                    ImageGenerationCapabilities {
                        minimum_width_pixels: 64,
                        maximum_width_pixels: 1_024,
                        minimum_height_pixels: 64,
                        maximum_height_pixels: 1_024,
                        dimension_multiple_pixels: 16,
                        maximum_steps: 64,
                        maximum_guidance_thousandths: 1_000,
                        output_mime_types: vec!["image/png".to_owned()],
                    },
                ))
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

#[tokio::test]
async fn should_preserve_a_pending_astro_ssd_prompt_cache_clear_across_model_replacement() {
    timeout(Duration::from_secs(1), async {
        let model_capabilities = ChatModelCapabilities {
            supports_reasoning: true,
            supports_tool_calls: true,
            has_vision: false,
            max_input_tokens: 1_024,
            max_output_tokens: 128,
            context_window: 2_048,
        };
        let mut previous_health_snapshot = WorkerHealthSnapshot::ready_with_model(
            "example/old-model".to_owned(),
            model_capabilities.clone(),
        );
        previous_health_snapshot.pending_astro_ssd_prompt_cache_clear =
            Some(PendingAstroSsdPromptCacheClear {
                model_id: Some("example/model-with-astro-ssd-prompt-cache-content".to_owned()),
            });

        let replacement_health_snapshot = WorkerHealthSnapshot::ready_with_replacement_model(
            "example/new-model".to_owned(),
            model_capabilities,
            1,
            &previous_health_snapshot,
        );

        assert_eq!(
            replacement_health_snapshot.pending_astro_ssd_prompt_cache_clear,
            previous_health_snapshot.pending_astro_ssd_prompt_cache_clear
        );
    })
    .await
    .expect("health replacement test should finish within one second");
}

fn completed_request_work(
    prompt_token_count: u32,
    restored_token_count: u32,
    prefill_elapsed_millis: u64,
    generated_token_count: u16,
    generation_elapsed_millis: u64,
) -> CompletedRequestWork {
    CompletedRequestWork {
        prompt_token_count,
        astro_ssd_prompt_cache_restored_token_count: restored_token_count,
        generated_token_count,
        prefill_elapsed_millis,
        generation_elapsed_millis,
    }
}

#[test]
fn should_report_session_speeds_as_total_tokens_over_total_time() {
    let mut serving_session_snapshot = ServingSessionSnapshot::empty();

    // 900 prefilled tokens in 1 s, then 2,700 in 9 s: 3,600 tokens over 10 s is 360 tok/s.
    // The mean of the two requests' own speeds (900 and 300) would say 600.
    serving_session_snapshot
        .record_completed_request(completed_request_work(1_000, 100, 1_000, 10, 1_000));
    serving_session_snapshot.record_prompt_work_reuse(WorkerPromptWorkReuse {
        target_eligible_token_count: 10_000,
        target_restored_token_count: 8_000,
    });
    serving_session_snapshot
        .record_completed_request(completed_request_work(3_000, 300, 9_000, 490, 9_000));

    assert_eq!(serving_session_snapshot.completed_request_count, 2);
    assert_eq!(serving_session_snapshot.total_prompt_token_count, 4_000);
    assert_eq!(
        serving_session_snapshot.total_reused_prompt_token_count,
        400
    );
    assert_eq!(
        serving_session_snapshot.average_prefill_tok_per_second,
        360.0
    );
    // 500 generated tokens over 10 s; the mean of 10 and about 54 tok/s would say 32.
    assert_eq!(
        serving_session_snapshot.average_generation_tok_per_second,
        50.0
    );
    assert_eq!(
        serving_session_snapshot.target_prompt_work_token_count,
        10_000
    );
    assert_eq!(
        serving_session_snapshot.target_reused_prompt_work_token_count,
        8_000
    );
}

#[test]
fn should_leave_a_speed_unchanged_when_a_request_has_nothing_to_measure_for_it() {
    let mut serving_session_snapshot = ServingSessionSnapshot::empty();
    serving_session_snapshot
        .record_completed_request(completed_request_work(1_000, 0, 2_000, 100, 4_000));

    // Fully restored prompt, no prefill time, no generated tokens, no generation time.
    serving_session_snapshot
        .record_completed_request(completed_request_work(5_000, 5_000, 3_000, 0, 6_000));
    serving_session_snapshot.record_completed_request(completed_request_work(5_000, 0, 0, 50, 0));
    // A restored count above the prompt length is capped at the prompt length.
    serving_session_snapshot
        .record_completed_request(completed_request_work(100, 900, 1_000, 0, 0));

    assert_eq!(serving_session_snapshot.completed_request_count, 4);
    assert_eq!(
        serving_session_snapshot.average_prefill_tok_per_second,
        500.0
    );
    assert_eq!(
        serving_session_snapshot.average_generation_tok_per_second,
        25.0
    );
    assert_eq!(
        serving_session_snapshot.total_reused_prompt_token_count,
        5_100
    );
}

#[test]
fn should_report_zero_speeds_before_any_request_is_measured() {
    let serving_session_snapshot = ServingSessionSnapshot::empty();
    assert_eq!(serving_session_snapshot.average_prefill_tok_per_second, 0.0);
    assert_eq!(
        serving_session_snapshot.average_generation_tok_per_second,
        0.0
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
    let mut previous_health_snapshot = astronomical_daemon::WorkerHealthSnapshot::ready_with_model(
        "first-model".to_owned(),
        capabilities.clone(),
    );
    previous_health_snapshot.mlx_memory_ceiling_bytes = 40_000;
    previous_health_snapshot
        .serving_session
        .record_completed_request(completed_request_work(1_000, 750, 25_000, 20, 1_000));

    let replacement_health_snapshot =
        astronomical_daemon::WorkerHealthSnapshot::ready_with_replacement_model(
            "second-model".to_owned(),
            capabilities,
            3_000,
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
