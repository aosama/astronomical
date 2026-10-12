use super::*;

#[tokio::test]
async fn should_trigger_internal_shutdown_when_shutdown_endpoint_is_called() {
    let shutdown_controller = ShutdownController::new();
    let mut shutdown_signal_receiver = shutdown_controller.subscribe();
    let application =
        build_application_with_shutdown(ScriptedExecutor::ready(Vec::new()), shutdown_controller);

    let response = post_shutdown(&application).await;
    assert_eq!(
        response.status(),
        StatusCode::ACCEPTED,
        "the shutdown endpoint must return HTTP 202"
    );
    let response_body = to_bytes(response.into_body(), 4 * 1024)
        .await
        .expect("the shutdown body should be readable");
    let response_json: serde_json::Value =
        serde_json::from_slice(&response_body).expect("the shutdown body should be JSON");
    assert_eq!(response_json["status"], "shutting_down");

    // The shutdown signal receiver must fire shortly after the POST.
    let shutdown_result = timeout(Duration::from_secs(2), shutdown_signal_receiver.changed())
        .await
        .expect("the shutdown signal must fire within 2 seconds");
    assert!(shutdown_result.is_ok(), "the shutdown watch must not error");
    assert!(
        *shutdown_signal_receiver.borrow(),
        "the shutdown signal must be true after the endpoint fires"
    );
}

#[test]
fn should_persist_shutdown_request_when_no_receiver_is_active() {
    let shutdown_controller = ShutdownController::new();

    assert!(shutdown_controller.request_shutdown());

    let shutdown_signal_receiver = shutdown_controller.subscribe();
    assert!(
        *shutdown_signal_receiver.borrow(),
        "a receiver subscribed after shutdown must observe the persisted request"
    );
}

#[tokio::test]
async fn should_keep_shutdown_endpoint_post_only() {
    let shutdown_controller = ShutdownController::new();
    let application =
        build_application_with_shutdown(ScriptedExecutor::ready(Vec::new()), shutdown_controller);
    let response = application
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/control/shutdown")
                .body(Body::empty())
                .expect("the GET request should be valid"),
        )
        .await
        .expect("the application should return a response");
    assert_eq!(
        response.status(),
        StatusCode::METHOD_NOT_ALLOWED,
        "the shutdown endpoint must be POST-only"
    );
}
