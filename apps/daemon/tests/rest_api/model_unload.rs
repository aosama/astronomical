//! `POST /v1/control/unload-model` through the real worker loop and the
//! daemon's stand-in worker process: a loaded model is unloaded by replacing
//! the worker with an empty one, nothing loaded is a no-op, and a reply being
//! written is never cut short.

use astronomical_daemon::{ChatGenerationExecutor, WorkerHandle, build_application};
use axum::{
    body::Body,
    http::{Method, Request, StatusCode},
};
use tower::ServiceExt;

use super::stand_in_worker::{
    DELAYED_MODEL_ID, delayed_generation_command, launch_stand_in_worker_application,
    receive_generation_completion, response_json, run_bounded_test,
};

async fn post_unload(application: &axum::Router) -> axum::response::Response {
    application
        .clone()
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/control/unload-model")
                .body(Body::empty())
                .expect("the unload request builds"),
        )
        .await
        .expect("the application answers")
}

#[tokio::test]
async fn should_unload_the_loaded_model_and_name_it() {
    run_bounded_test(async {
        let test_context = launch_stand_in_worker_application().await;
        let mut generation_events = test_context
            .worker_handle
            .start_chat_generation(delayed_generation_command(61))
            .await
            .expect("generation should start");
        receive_generation_completion(&mut generation_events).await;

        let unload_response = post_unload(&test_context.application).await;
        assert_eq!(unload_response.status(), StatusCode::OK);
        let unload_document = response_json(unload_response).await;
        assert_eq!(unload_document["status"], "unloaded");
        assert_eq!(unload_document["model_id"], DELAYED_MODEL_ID);

        // The empty worker holds nothing, so a second unload is a no-op.
        let second_response = post_unload(&test_context.application).await;
        assert_eq!(second_response.status(), StatusCode::OK);
        assert_eq!(
            response_json(second_response).await["status"],
            "nothing_loaded"
        );

        // The fresh worker still serves: the next request loads its model.
        let mut next_generation_events = test_context
            .worker_handle
            .start_chat_generation(delayed_generation_command(62))
            .await
            .expect("a request after the unload should start");
        receive_generation_completion(&mut next_generation_events).await;
        test_context.shutdown().await;
    })
    .await;
}

#[tokio::test]
async fn should_answer_nothing_loaded_before_any_model_has_loaded() {
    run_bounded_test(async {
        let test_context = launch_stand_in_worker_application().await;
        let unload_response = post_unload(&test_context.application).await;
        assert_eq!(unload_response.status(), StatusCode::OK);
        let unload_document = response_json(unload_response).await;
        assert_eq!(unload_document["status"], "nothing_loaded");
        assert!(unload_document["model_id"].is_null());
        test_context.shutdown().await;
    })
    .await;
}

#[tokio::test]
async fn should_refuse_while_a_reply_is_being_written_and_keep_it_going() {
    run_bounded_test(async {
        let test_context = launch_stand_in_worker_application().await;
        let mut generation_events = test_context
            .worker_handle
            .start_chat_generation(delayed_generation_command(63))
            .await
            .expect("generation should start");

        let unload_response = post_unload(&test_context.application).await;
        assert_eq!(unload_response.status(), StatusCode::CONFLICT);
        assert_eq!(response_json(unload_response).await["status"], "busy");

        // The reply finishes as if nothing had been asked.
        receive_generation_completion(&mut generation_events).await;
        test_context.shutdown().await;
    })
    .await;
}

#[tokio::test]
async fn should_not_offer_unload_without_live_worker_control() {
    run_bounded_test(async {
        let application = build_application(WorkerHandle::unavailable());
        let unload_response = post_unload(&application).await;
        assert_eq!(unload_response.status(), StatusCode::NOT_FOUND);
    })
    .await;
}

#[tokio::test]
async fn should_answer_unavailable_when_the_worker_has_stopped() {
    run_bounded_test(async {
        let test_context = launch_stand_in_worker_application().await;
        test_context
            .worker_handle
            .clone()
            .shutdown()
            .await
            .expect("the stand-in worker should stop");
        let unload_response = post_unload(&test_context.application).await;
        assert_eq!(unload_response.status(), StatusCode::SERVICE_UNAVAILABLE);
    })
    .await;
}
