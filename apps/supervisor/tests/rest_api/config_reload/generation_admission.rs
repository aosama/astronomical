//! Ensures configuration control remains responsive after a request reserves queue capacity.

use super::*;

#[tokio::test]
async fn should_release_configuration_transition_after_request_queue_admission() {
    let config_home_directory = tempfile::tempdir()
        .expect("a config home should be created")
        .keep();
    write_config_file(&config_home_directory, r#"{}"#);
    let admission_started = Arc::new(tokio::sync::Notify::new());
    let release_admission = Arc::new(tokio::sync::Notify::new());
    let delayed_executor = DelayedAdmissionExecutor {
        delegate: ScriptedExecutor::ready(Vec::new()),
        admission_started: Arc::clone(&admission_started),
        release_admission: Arc::clone(&release_admission),
    };
    let application = build_development_application_with_reload(
        delayed_executor,
        Arc::new(RwLock::new(sample_resolved_config())),
        config_home_directory,
    );

    let request_application = application.clone();
    let request_task = tokio::spawn(async move {
        request_application
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(format!(
                        r#"{{"model":"{}","messages":[{{"role":"user","content":"hello"}}],"stream":true}}"#,
                        crate::common::MODEL_ID
                    )))
                    .expect("the chat request should be valid"),
            )
            .await
            .expect("the chat request should receive a response")
    });
    timeout(Duration::from_secs(2), admission_started.notified())
        .await
        .expect("generation admission should begin");

    let reload_application = application.clone();
    let reload_task = tokio::spawn(async move { post_config_reload(&reload_application).await });
    timeout(Duration::from_secs(2), reload_task)
        .await
        .expect("reload should not wait for queued generation execution")
        .expect("the reload task should finish");

    release_admission.notify_one();
    let request_response = request_task.await.expect("the request task should finish");
    assert_eq!(request_response.status(), StatusCode::OK);
}

struct DelayedAdmissionExecutor {
    delegate: ScriptedExecutor,
    admission_started: Arc<tokio::sync::Notify>,
    release_admission: Arc<tokio::sync::Notify>,
}

impl ChatGenerationExecutor for DelayedAdmissionExecutor {
    fn start_chat_generation(
        &self,
        generation_command: astronomical_ipc_protocol::ChatGenerationCommand,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<
                    Output = Result<
                        tokio::sync::mpsc::Receiver<
                            astronomical_supervisor::ChatGenerationStreamEvent,
                        >,
                        astronomical_supervisor::GenerationStartError,
                    >,
                > + Send
                + '_,
        >,
    > {
        Box::pin(async move {
            self.admission_started.notify_one();
            self.release_admission.notified().await;
            self.delegate
                .start_chat_generation(generation_command)
                .await
        })
    }

    fn worker_health_snapshot(&self) -> astronomical_supervisor::WorkerHealthSnapshot {
        self.delegate.worker_health_snapshot()
    }
}
