use super::*;

const MAXIMUM_RESPONSE_BYTES: usize = 1024 * 1024;

fn valid_document() -> serde_json::Value {
    serde_json::json!({
        "$schema": "./astronomical-config.schema.json",
        "schema_version": 1,
        "runtime": { "model_directories": [] }
    })
}

fn application_with_config(
    home_directory: &std::path::Path,
) -> (
    axum::Router,
    Arc<RwLock<astronomical_daemon::ResolvedRuntimeConfig>>,
) {
    let reloadable_config = Arc::new(RwLock::new(sample_resolved_config()));
    let application = build_development_application_with_reload(
        ScriptedExecutor::ready(Vec::new()),
        reloadable_config.clone(),
        home_directory.to_path_buf(),
    );
    (application, reloadable_config)
}

async fn json_of(response: axum::response::Response) -> serde_json::Value {
    let response_body = to_bytes(response.into_body(), MAXIMUM_RESPONSE_BYTES)
        .await
        .expect("the response body should be readable");
    serde_json::from_slice(&response_body).expect("the response should be JSON")
}

async fn get_config(application: &axum::Router) -> axum::response::Response {
    application
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/config")
                .body(Body::empty())
                .expect("the request should be valid"),
        )
        .await
        .expect("the application should respond")
}

async fn send_json(
    application: &axum::Router,
    method: &str,
    uri: &str,
    body: &serde_json::Value,
) -> axum::response::Response {
    application
        .clone()
        .oneshot(
            Request::builder()
                .method(method)
                .uri(uri)
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(body.to_string()))
                .expect("the request should be valid"),
        )
        .await
        .expect("the application should respond")
}

fn config_file_text(home_directory: &std::path::Path) -> String {
    std::fs::read_to_string(home_directory.join(".astronomical-dev").join("config.json"))
        .expect("the config file should read")
}

#[tokio::test]
async fn should_serve_the_document_its_digest_and_the_schema_describing_it() {
    let home_directory = tempfile::tempdir().expect("a temp directory");
    write_raw_config_file(home_directory.path(), &valid_document().to_string());
    let (application, _reloadable_config) = application_with_config(home_directory.path());

    let response = get_config(&application).await;

    assert_eq!(response.status(), StatusCode::OK);
    let response_json = json_of(response).await;
    assert_eq!(response_json["document"], valid_document());
    assert_eq!(response_json["read_error"], serde_json::Value::Null);
    assert_eq!(
        response_json["document_digest"]
            .as_str()
            .expect("a digest is present")
            .len(),
        64
    );
    assert_eq!(
        response_json["schema"]["$defs"]["chunking"]["properties"]["experimental_neural_engine_prefill_enabled"]
            ["default"],
        false
    );
}

#[tokio::test]
async fn should_serve_a_read_error_for_a_broken_file_instead_of_failing() {
    let home_directory = tempfile::tempdir().expect("a temp directory");
    write_raw_config_file(home_directory.path(), "{ not json");
    let (application, _reloadable_config) = application_with_config(home_directory.path());

    let response = get_config(&application).await;

    assert_eq!(response.status(), StatusCode::OK);
    let response_json = json_of(response).await;
    assert_eq!(response_json["document"], serde_json::Value::Null);
    assert!(response_json["read_error"].is_string());
}

#[tokio::test]
async fn should_save_a_valid_document_and_reload_it() {
    let home_directory = tempfile::tempdir().expect("a temp directory");
    write_raw_config_file(home_directory.path(), &valid_document().to_string());
    let (application, _reloadable_config) = application_with_config(home_directory.path());
    let digest = json_of(get_config(&application).await).await["document_digest"].clone();
    let mut edited_document = valid_document();
    edited_document["diagnostics"] = serde_json::json!({ "log_level": "info" });

    let response = send_json(
        &application,
        "PUT",
        "/v1/config",
        &serde_json::json!({ "document": edited_document, "expected_document_digest": digest }),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    let response_json = json_of(response).await;
    assert_eq!(response_json["status"], "saved");
    assert!(response_json["reload_http_status"].is_u64());
    assert!(config_file_text(home_directory.path()).contains("\"log_level\": \"info\""));
}

#[tokio::test]
async fn should_refuse_an_invalid_document_with_a_message_and_leave_the_file_alone() {
    let home_directory = tempfile::tempdir().expect("a temp directory");
    let original = valid_document().to_string();
    write_raw_config_file(home_directory.path(), &original);
    let (application, _reloadable_config) = application_with_config(home_directory.path());
    let digest = json_of(get_config(&application).await).await["document_digest"].clone();
    let mut invalid_document = valid_document();
    invalid_document["diagnostics"] = serde_json::json!({ "retained_log_files": 0 });

    let response = send_json(
        &application,
        "PUT",
        "/v1/config",
        &serde_json::json!({ "document": invalid_document, "expected_document_digest": digest }),
    )
    .await;

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let response_json = json_of(response).await;
    assert_eq!(response_json["status"], "invalid_config");
    assert!(response_json["message"].is_string());
    assert_eq!(config_file_text(home_directory.path()), original);
}

#[tokio::test]
async fn should_refuse_a_save_based_on_a_stale_digest() {
    let home_directory = tempfile::tempdir().expect("a temp directory");
    write_raw_config_file(home_directory.path(), &valid_document().to_string());
    let (application, _reloadable_config) = application_with_config(home_directory.path());
    let stale_digest = json_of(get_config(&application).await).await["document_digest"].clone();
    let hand_edit = valid_document().to_string() + "\n";
    write_raw_config_file(home_directory.path(), &hand_edit);

    let response = send_json(
        &application,
        "PUT",
        "/v1/config",
        &serde_json::json!({
            "document": valid_document(),
            "expected_document_digest": stale_digest
        }),
    )
    .await;

    assert_eq!(response.status(), StatusCode::CONFLICT);
    assert_eq!(json_of(response).await["status"], "changed_elsewhere");
    assert_eq!(config_file_text(home_directory.path()), hand_edit);
}

#[tokio::test]
async fn should_reset_to_defaults_keeping_the_previous_file() {
    let home_directory = tempfile::tempdir().expect("a temp directory");
    let mut original = valid_document();
    original["diagnostics"] = serde_json::json!({ "log_level": "debug" });
    write_raw_config_file(home_directory.path(), &original.to_string());
    let (application, _reloadable_config) = application_with_config(home_directory.path());
    let digest = json_of(get_config(&application).await).await["document_digest"].clone();

    let response = send_json(
        &application,
        "POST",
        "/v1/config/reset",
        &serde_json::json!({ "expected_document_digest": digest }),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(json_of(response).await["status"], "saved");
    assert!(!config_file_text(home_directory.path()).contains("log_level"));
    assert!(
        std::fs::read_to_string(
            home_directory
                .path()
                .join(".astronomical-dev")
                .join("config.previous.json")
        )
        .expect("the previous file was kept")
        .contains("debug")
    );
}

#[tokio::test]
async fn should_refuse_a_save_request_with_an_unknown_field() {
    let home_directory = tempfile::tempdir().expect("a temp directory");
    write_raw_config_file(home_directory.path(), &valid_document().to_string());
    let (application, _reloadable_config) = application_with_config(home_directory.path());

    let response = send_json(
        &application,
        "PUT",
        "/v1/config",
        &serde_json::json!({ "document": valid_document(), "force": true }),
    )
    .await;

    assert!(response.status().is_client_error());
}

async fn request_with_browser_headers(
    application: &axum::Router,
    method: &str,
    uri: &str,
    host: &str,
    origin: &str,
    body: &serde_json::Value,
) -> axum::response::Response {
    application
        .clone()
        .oneshot(
            Request::builder()
                .method(method)
                .uri(uri)
                .header(header::HOST, host)
                .header(header::ORIGIN, origin)
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(body.to_string()))
                .expect("the request should be valid"),
        )
        .await
        .expect("the application should respond")
}

#[tokio::test]
async fn should_refuse_a_rebound_page_on_every_configuration_endpoint() {
    let home_directory = tempfile::tempdir().expect("a temp directory");
    let original = valid_document().to_string();
    write_raw_config_file(home_directory.path(), &original);
    let (application, _reloadable_config) = application_with_config(home_directory.path());
    let digest = json_of(get_config(&application).await).await["document_digest"].clone();
    let save_body = serde_json::json!({
        "document": valid_document(),
        "expected_document_digest": digest
    });
    let reset_body = serde_json::json!({ "expected_document_digest": digest });

    for (method, uri, body) in [
        ("PUT", "/v1/config", &save_body),
        ("POST", "/v1/config/reset", &reset_body),
    ] {
        let response = request_with_browser_headers(
            &application,
            method,
            uri,
            "attacker.example:6733",
            "http://attacker.example:6733",
            body,
        )
        .await;
        assert_eq!(response.status(), StatusCode::FORBIDDEN, "{method} {uri}");
    }
    let read_response = application
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/config")
                .header(header::HOST, "attacker.example:6733")
                .body(Body::empty())
                .expect("the request should be valid"),
        )
        .await
        .expect("the application should respond");

    assert_eq!(read_response.status(), StatusCode::FORBIDDEN);
    assert_eq!(config_file_text(home_directory.path()), original);
}

#[tokio::test]
async fn should_refuse_another_local_web_server_and_accept_the_instances_own_page() {
    let home_directory = tempfile::tempdir().expect("a temp directory");
    write_raw_config_file(home_directory.path(), &valid_document().to_string());
    let (application, _reloadable_config) = application_with_config(home_directory.path());
    let digest = json_of(get_config(&application).await).await["document_digest"].clone();
    let save_body = serde_json::json!({
        "document": valid_document(),
        "expected_document_digest": digest
    });

    let foreign_response = request_with_browser_headers(
        &application,
        "PUT",
        "/v1/config",
        "127.0.0.1:6733",
        "http://localhost:3000",
        &save_body,
    )
    .await;
    assert_eq!(foreign_response.status(), StatusCode::FORBIDDEN);

    let own_page_response = request_with_browser_headers(
        &application,
        "PUT",
        "/v1/config",
        "127.0.0.1:6733",
        "http://127.0.0.1:6733",
        &save_body,
    )
    .await;
    assert_eq!(own_page_response.status(), StatusCode::OK);
}
