use crate::rest_api::http_exchange_fixtures::get_response;
use std::time::Duration;

use astronomical_daemon::{ChatGenerationExecutor, WorkerHandle, build_application};
use axum::{
    body::Body,
    http::{Method, Request, StatusCode},
};
use tokio::time::{Instant, sleep, timeout};
use tower::ServiceExt;

use super::stand_in_worker::{
    delayed_generation_command, launch_stand_in_worker_application, receive_generation_completion,
    response_json, run_bounded_test,
};

#[tokio::test]
async fn should_clear_the_entire_astro_ssd_prompt_cache_while_the_worker_is_idle() {
    run_bounded_test(async {
        let test_context = launch_stand_in_worker_application().await;
        eprintln!("[astro-ssd-prompt-cache-clear-rest] deleting the global astro SSD prompt cache while idle");

        let clear_response =
            delete_astro_ssd_prompt_cache(&test_context.application, "/v1/ssd-prompt-cache").await;
        assert_eq!(clear_response.status(), StatusCode::OK);
        let clear_document = response_json(clear_response).await;
        assert_eq!(clear_document["status"], "cleared");
        assert_eq!(clear_document["model_id"], serde_json::Value::Null);
        assert_eq!(clear_document["blocks_removed"], 3);
        assert_eq!(clear_document["bytes_freed"], 4_096);
        wait_for_zeroed_astro_ssd_prompt_cache_stats(&test_context.application).await;

        test_context.shutdown().await;
    })
    .await;
}

#[tokio::test]
async fn should_clear_one_model_astro_ssd_prompt_cache_by_its_model_id() {
    run_bounded_test(async {
        let test_context = launch_stand_in_worker_application().await;

        let clear_response = delete_astro_ssd_prompt_cache(
            &test_context.application,
            "/v1/ssd-prompt-cache?model=astronomical%2Frequested-model",
        )
        .await;
        assert_eq!(clear_response.status(), StatusCode::OK);
        let clear_document = response_json(clear_response).await;
        assert_eq!(clear_document["model_id"], "astronomical/requested-model");

        test_context.shutdown().await;
    })
    .await;
}

#[tokio::test]
async fn should_queue_only_the_newest_astro_ssd_prompt_cache_clear_until_generation_finishes() {
    run_bounded_test(async {
        let test_context = launch_stand_in_worker_application().await;
        eprintln!(
            "[astro-ssd-prompt-cache-clear-rest] starting the Romeo and Juliet generation journey"
        );
        let mut generation_events = test_context
            .worker_handle
            .start_chat_generation(delayed_generation_command(41))
            .await
            .expect("generation should start");

        let global_clear_response =
            delete_astro_ssd_prompt_cache(&test_context.application, "/v1/ssd-prompt-cache").await;
        assert_eq!(global_clear_response.status(), StatusCode::ACCEPTED);
        let scoped_clear_response = delete_astro_ssd_prompt_cache(
            &test_context.application,
            "/v1/ssd-prompt-cache?model=astronomical%2Frequested-model",
        )
        .await;
        assert_eq!(scoped_clear_response.status(), StatusCode::ACCEPTED);

        let queued_stats = get_astro_ssd_prompt_cache_stats(&test_context.application).await;
        assert_eq!(
            queued_stats["pending_astro_ssd_prompt_cache_clear"]["model_id"],
            "astronomical/requested-model"
        );

        timeout(Duration::from_secs(2), generation_events.recv())
            .await
            .expect("generation should finish before the timeout")
            .expect("generation should emit completion");
        wait_for_pending_clear_to_finish(&test_context.application).await;

        test_context.shutdown().await;
    })
    .await;
}

#[tokio::test]
async fn should_wait_for_a_queued_generation_before_applying_the_astro_ssd_prompt_cache_clear() {
    run_bounded_test(async {
        let test_context = launch_stand_in_worker_application().await;
        let mut first_generation_events = test_context
            .worker_handle
            .start_chat_generation(delayed_generation_command(51))
            .await
            .expect("first generation should start");
        let queued_worker_handle = test_context.worker_handle.clone();
        let queued_generation_task = tokio::spawn(async move {
            queued_worker_handle
                .start_chat_generation(delayed_generation_command(52))
                .await
        });

        let clear_response = delete_astro_ssd_prompt_cache(&test_context.application, "/v1/ssd-prompt-cache").await;
        assert_eq!(clear_response.status(), StatusCode::ACCEPTED);
        receive_generation_completion(&mut first_generation_events).await;
        let mut queued_generation_events = timeout(Duration::from_secs(2), queued_generation_task)
            .await
            .expect("queued generation should start after the first")
            .expect("queued generation task should not panic")
            .expect("queued generation should be accepted");

        let astro_ssd_prompt_cache_stats_during_queued_generation = get_astro_ssd_prompt_cache_stats(&test_context.application).await;
        assert!(
            !astro_ssd_prompt_cache_stats_during_queued_generation["pending_astro_ssd_prompt_cache_clear"].is_null(),
            "astro SSD prompt cache clear must remain pending while the queued generation runs"
        );
        receive_generation_completion(&mut queued_generation_events).await;
        wait_for_pending_clear_to_finish(&test_context.application).await;

        test_context.shutdown().await;
    })
    .await;
}

#[tokio::test]
async fn should_reject_a_model_id_that_can_escape_the_astro_ssd_prompt_cache_root() {
    run_bounded_test(async {
        let test_context = launch_stand_in_worker_application().await;

        let clear_response = delete_astro_ssd_prompt_cache(
            &test_context.application,
            "/v1/ssd-prompt-cache?model=..%2Foutside",
        )
        .await;
        assert_eq!(clear_response.status(), StatusCode::BAD_REQUEST);
        let empty_model_response =
            delete_astro_ssd_prompt_cache(&test_context.application, "/v1/ssd-prompt-cache?model=")
                .await;
        assert_eq!(empty_model_response.status(), StatusCode::BAD_REQUEST);

        test_context.shutdown().await;
    })
    .await;
}

#[tokio::test]
async fn should_not_offer_astro_ssd_prompt_cache_clear_without_live_worker_control() {
    run_bounded_test(async {
        let application = build_application(WorkerHandle::unavailable());

        let clear_response =
            delete_astro_ssd_prompt_cache(&application, "/v1/ssd-prompt-cache").await;

        assert_eq!(clear_response.status(), StatusCode::NOT_FOUND);
    })
    .await;
}

#[tokio::test]
async fn should_return_service_unavailable_when_the_worker_has_stopped() {
    run_bounded_test(async {
        let test_context = launch_stand_in_worker_application().await;
        test_context
            .worker_handle
            .clone()
            .shutdown()
            .await
            .expect("worker fixture should stop");

        let clear_response =
            delete_astro_ssd_prompt_cache(&test_context.application, "/v1/ssd-prompt-cache").await;

        assert_eq!(clear_response.status(), StatusCode::SERVICE_UNAVAILABLE);
    })
    .await;
}

#[tokio::test]
async fn should_reject_a_worker_acknowledgement_for_a_different_model_scope() {
    run_bounded_test(async {
        let test_context = launch_stand_in_worker_application().await;

        let clear_response = delete_astro_ssd_prompt_cache(
            &test_context.application,
            "/v1/ssd-prompt-cache?model=astronomical%2Fmismatched-clear-model",
        )
        .await;

        assert_eq!(clear_response.status(), StatusCode::SERVICE_UNAVAILABLE);
        test_context.shutdown().await;
    })
    .await;
}

async fn delete_astro_ssd_prompt_cache(
    application: &axum::Router,
    request_uri: &str,
) -> axum::response::Response {
    application
        .clone()
        .oneshot(
            Request::builder()
                .method(Method::DELETE)
                .uri(request_uri)
                .body(Body::empty())
                .expect("astro SSD prompt cache clear request should be valid"),
        )
        .await
        .expect("application should return an astro SSD prompt cache clear response")
}

async fn get_astro_ssd_prompt_cache_stats(application: &axum::Router) -> serde_json::Value {
    let stats_response = get_response(application, "/v1/ssd-prompt-cache/stats").await;
    response_json(stats_response).await
}

async fn wait_for_pending_clear_to_finish(application: &axum::Router) {
    let clear_deadline = Instant::now() + Duration::from_secs(2);
    loop {
        let astro_ssd_prompt_cache_stats = get_astro_ssd_prompt_cache_stats(application).await;
        if astro_ssd_prompt_cache_stats["pending_astro_ssd_prompt_cache_clear"].is_null() {
            return;
        }
        assert!(
            Instant::now() < clear_deadline,
            "queued clear should finish"
        );
        sleep(Duration::from_millis(10)).await;
    }
}

async fn wait_for_zeroed_astro_ssd_prompt_cache_stats(application: &axum::Router) {
    let stats_deadline = Instant::now() + Duration::from_secs(2);
    loop {
        let astro_ssd_prompt_cache_stats = get_astro_ssd_prompt_cache_stats(application).await;
        if astro_ssd_prompt_cache_stats["astro_ssd_prompt_cache_maximum_size_bytes"]
            == 50_000_000_000_u64
        {
            assert_eq!(
                astro_ssd_prompt_cache_stats["astro_ssd_prompt_cache_sequence_state_block_count"],
                0
            );
            assert_eq!(
                astro_ssd_prompt_cache_stats["astro_ssd_prompt_cache_total_size_bytes"],
                0
            );
            return;
        }
        assert!(
            Instant::now() < stats_deadline,
            "cleared stats should arrive"
        );
        sleep(Duration::from_millis(10)).await;
    }
}
