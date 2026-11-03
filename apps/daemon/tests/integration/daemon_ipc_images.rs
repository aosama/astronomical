//! Daemon IPC image journeys: routing `ImageGenerate` requests to the worker
//! executor over the real socket, and rejecting the ones that cannot succeed
//! before they reach it. The CLI reaches image generation only this way.

use std::{
    path::Path,
    sync::{Arc, RwLock},
};

use astronomical_daemon::{
    ImageGenerationExecutionError, ImageGenerationOutput, WorkerHealthSnapshot, WorkerHealthStatus,
};
use astronomical_ipc_protocol::{
    DaemonIpcClient, DaemonListedModel, DaemonRequest, DaemonResponse, GeneratedImage,
    ImageGenerateRequest, ImageGenerationFailureReason, ImageGenerationResultMetadata, RequestId,
};
use tokio::time::timeout;

use crate::support::daemon_ipc::{
    HANDSHAKE_TEST_TIMEOUT, StubGenerationExecutor, fresh_instance_state_directory,
    ipc_chat_discovered_model, ipc_image_discovered_model, ipc_runtime_config, ready_stub_executor,
    ready_stub_executor_with_image, start_stub_daemon_ipc_service_with_runtime_config,
};

const IMAGE_MODEL_ID: &str = "test-flux";
const CHAT_MODEL_ID: &str = "test-chatter";

fn completed_image(seed: u64) -> Result<ImageGenerationOutput, ImageGenerationExecutionError> {
    Ok(ImageGenerationOutput {
        generated_image: GeneratedImage {
            mime_type: "image/png".to_owned(),
            encoded_bytes: vec![0x89, b'P', b'N', b'G'],
        },
        result_metadata: ImageGenerationResultMetadata {
            width_pixels: 1_024,
            height_pixels: 576,
            steps: 4,
            guidance_thousandths: 1_000,
            seed,
            elapsed_millis: 1_500,
        },
    })
}

fn image_request(
    model: &str,
    width_pixels: u32,
    height_pixels: u32,
    seed: Option<u64>,
) -> DaemonRequest {
    DaemonRequest::ImageGenerate(ImageGenerateRequest {
        model: model.to_owned(),
        prompt: "a lighthouse at dusk".to_owned(),
        width_pixels,
        height_pixels,
        seed,
    })
}

/// Sends one request on a fresh connection and returns the daemon's single
/// terminal reply.
async fn send_and_receive(socket_path: &Path, daemon_request: &DaemonRequest) -> DaemonResponse {
    let mut daemon_client = DaemonIpcClient::connect(socket_path.to_path_buf())
        .await
        .expect("the client should connect to the running daemon IPC service");
    timeout(
        HANDSHAKE_TEST_TIMEOUT,
        daemon_client.send_request(daemon_request),
    )
    .await
    .expect("the request send should finish inside the test timeout")
    .expect("the request should transmit");
    timeout(HANDSHAKE_TEST_TIMEOUT, daemon_client.next_response())
        .await
        .expect("the reply should arrive inside the test timeout")
        .expect("the reply read should not fail at the transport layer")
        .expect("the daemon should answer the request")
}

/// Starts a daemon that knows one image model and one chat model, sends the
/// requests in order, and returns every reply plus the executor, so a test
/// can also see what reached the worker.
async fn serve_requests(
    test_name: &str,
    stub_executor: Arc<StubGenerationExecutor>,
    daemon_requests: Vec<DaemonRequest>,
) -> (Vec<DaemonResponse>, Arc<StubGenerationExecutor>) {
    let state_directory = fresh_instance_state_directory(test_name);
    let reloadable_config = Arc::new(RwLock::new(ipc_runtime_config(
        vec![
            ipc_image_discovered_model(IMAGE_MODEL_ID),
            ipc_chat_discovered_model(CHAT_MODEL_ID),
        ],
        &[CHAT_MODEL_ID],
    )));
    let daemon_ipc_service = start_stub_daemon_ipc_service_with_runtime_config(
        &state_directory,
        Arc::clone(&stub_executor),
        Some(reloadable_config),
    )
    .await;
    let mut daemon_replies = Vec::new();
    for daemon_request in &daemon_requests {
        daemon_replies
            .push(send_and_receive(daemon_ipc_service.socket_path(), daemon_request).await);
    }
    timeout(HANDSHAKE_TEST_TIMEOUT, daemon_ipc_service.shutdown())
        .await
        .expect("the daemon IPC service shutdown should finish inside the test timeout")
        .expect("the daemon IPC service should shut down cleanly");
    let _ = std::fs::remove_dir_all(&state_directory);
    (daemon_replies, stub_executor)
}

fn rejection_reason_of(daemon_reply: &DaemonResponse) -> String {
    let DaemonResponse::GenerationRejected { reason } = daemon_reply else {
        panic!("the daemon should have rejected the request, not answered {daemon_reply:?}");
    };
    reason.clone()
}

#[tokio::test]
async fn should_route_an_image_request_to_the_executor_and_return_the_png() {
    let stub_executor = ready_stub_executor_with_image(CHAT_MODEL_ID, completed_image(42));

    let (daemon_replies, stub_executor) = serve_requests(
        "image-routed",
        stub_executor,
        vec![image_request(IMAGE_MODEL_ID, 1_024, 576, Some(42))],
    )
    .await;

    let DaemonResponse::ImageGenerationCompleted {
        model,
        image,
        metadata,
    } = &daemon_replies[0]
    else {
        panic!(
            "the daemon should have completed the image, not answered {:?}",
            daemon_replies[0]
        );
    };
    assert_eq!(model, IMAGE_MODEL_ID);
    assert_eq!(image.mime_type, "image/png");
    assert_eq!(image.encoded_bytes, vec![0x89, b'P', b'N', b'G']);
    assert_eq!(metadata.seed, 42);
    let received_commands = stub_executor.logged_image_commands();
    assert_eq!(received_commands.len(), 1);
    assert_eq!(received_commands[0].request_id, RequestId::new(1));
    assert_eq!(received_commands[0].model, IMAGE_MODEL_ID);
    assert_eq!(received_commands[0].prompt, "a lighthouse at dusk");
    assert_eq!(
        received_commands[0].settings.steps, 4,
        "the family's own step count applies"
    );
    assert_eq!(received_commands[0].settings.guidance_thousandths, 1_000);
    assert_eq!(received_commands[0].settings.seed, 42);
}

#[tokio::test]
async fn should_draw_a_different_seed_for_each_request_that_carries_none() {
    let stub_executor = ready_stub_executor_with_image(CHAT_MODEL_ID, completed_image(1));

    let (_daemon_replies, stub_executor) = serve_requests(
        "image-seeds",
        stub_executor,
        vec![
            image_request(IMAGE_MODEL_ID, 512, 512, None),
            image_request(IMAGE_MODEL_ID, 512, 512, None),
        ],
    )
    .await;

    let received_commands = stub_executor.logged_image_commands();
    assert_eq!(received_commands.len(), 2);
    assert_ne!(
        received_commands[0].settings.seed,
        received_commands[1].settings.seed
    );
}

#[tokio::test]
async fn should_reject_a_chat_model_without_admitting_the_request() {
    let stub_executor = ready_stub_executor_with_image(CHAT_MODEL_ID, completed_image(1));

    let (daemon_replies, stub_executor) = serve_requests(
        "image-chat-model",
        stub_executor,
        vec![image_request(CHAT_MODEL_ID, 512, 512, None)],
    )
    .await;

    let rejection_reason = rejection_reason_of(&daemon_replies[0]);
    assert!(
        rejection_reason.contains("does not support text-to-image"),
        "{rejection_reason}"
    );
    assert!(
        stub_executor.logged_image_commands().is_empty(),
        "a doomed request must not be admitted"
    );
}

#[tokio::test]
async fn should_reject_an_unknown_model_and_name_the_installed_ones() {
    let stub_executor = ready_stub_executor_with_image(CHAT_MODEL_ID, completed_image(1));

    let (daemon_replies, stub_executor) = serve_requests(
        "image-unknown-model",
        stub_executor,
        vec![image_request("test-fluxx", 512, 512, None)],
    )
    .await;

    let rejection_reason = rejection_reason_of(&daemon_replies[0]);
    assert!(
        rejection_reason.contains("is unknown"),
        "{rejection_reason}"
    );
    assert!(rejection_reason.contains("test-flux"), "{rejection_reason}");
    assert!(stub_executor.logged_image_commands().is_empty());
}

#[tokio::test]
async fn should_reject_sizes_outside_the_model_envelope_before_admission() {
    let stub_executor = ready_stub_executor_with_image(CHAT_MODEL_ID, completed_image(1));
    let rejected_sizes = [
        (1_000, 512, "a multiple of 16"),
        (512, 2_048, "at most 1024"),
        (32, 512, "at least 64"),
    ];
    let mut daemon_requests = Vec::new();
    for (width_pixels, height_pixels, _expected_text) in rejected_sizes {
        daemon_requests.push(image_request(
            IMAGE_MODEL_ID,
            width_pixels,
            height_pixels,
            None,
        ));
    }

    let (daemon_replies, stub_executor) =
        serve_requests("image-sizes", stub_executor, daemon_requests).await;

    for (daemon_reply, (_width, _height, expected_text)) in
        daemon_replies.iter().zip(rejected_sizes)
    {
        let rejection_reason = rejection_reason_of(daemon_reply);
        assert!(
            rejection_reason.contains(expected_text),
            "{rejection_reason}"
        );
    }
    assert!(
        stub_executor.logged_image_commands().is_empty(),
        "no out-of-envelope size may reach the worker"
    );
}

#[tokio::test]
async fn should_accept_the_edges_of_the_model_envelope() {
    let stub_executor = ready_stub_executor_with_image(CHAT_MODEL_ID, completed_image(1));

    let (daemon_replies, stub_executor) = serve_requests(
        "image-edges",
        stub_executor,
        vec![
            image_request(IMAGE_MODEL_ID, 64, 64, None),
            image_request(IMAGE_MODEL_ID, 1_024, 1_024, None),
        ],
    )
    .await;

    for daemon_reply in &daemon_replies {
        assert!(
            matches!(
                daemon_reply,
                DaemonResponse::ImageGenerationCompleted { .. }
            ),
            "{daemon_reply:?}"
        );
    }
    assert_eq!(stub_executor.logged_image_commands().len(), 2);
}

#[tokio::test]
async fn should_reject_a_blank_prompt_before_admission() {
    let stub_executor = ready_stub_executor_with_image(CHAT_MODEL_ID, completed_image(1));
    let blank_prompt_request = DaemonRequest::ImageGenerate(ImageGenerateRequest {
        model: IMAGE_MODEL_ID.to_owned(),
        prompt: "   ".to_owned(),
        width_pixels: 512,
        height_pixels: 512,
        seed: None,
    });

    let (daemon_replies, stub_executor) = serve_requests(
        "image-blank-prompt",
        stub_executor,
        vec![blank_prompt_request],
    )
    .await;

    assert!(matches!(
        daemon_replies[0],
        DaemonResponse::GenerationRejected { .. }
    ));
    assert!(stub_executor.logged_image_commands().is_empty());
}

#[tokio::test]
async fn should_reject_while_the_worker_is_still_loading() {
    let mut stub_executor = ready_stub_executor_with_image(CHAT_MODEL_ID, completed_image(1));
    Arc::get_mut(&mut stub_executor)
        .expect("the stub executor should be uniquely owned here")
        .health_snapshot = WorkerHealthSnapshot::unavailable(WorkerHealthStatus::Loading);

    let (daemon_replies, stub_executor) = serve_requests(
        "image-worker-loading",
        stub_executor,
        vec![image_request(IMAGE_MODEL_ID, 512, 512, None)],
    )
    .await;

    assert!(rejection_reason_of(&daemon_replies[0]).contains("not ready"));
    assert!(stub_executor.logged_image_commands().is_empty());
}

#[tokio::test]
async fn should_pass_a_worker_failure_after_admission_to_the_client() {
    let stub_executor = ready_stub_executor_with_image(
        CHAT_MODEL_ID,
        Err(ImageGenerationExecutionError::WorkerFailure(
            ImageGenerationFailureReason::EngineBusy,
        )),
    );

    let (daemon_replies, stub_executor) = serve_requests(
        "image-worker-failure",
        stub_executor,
        vec![image_request(IMAGE_MODEL_ID, 512, 512, None)],
    )
    .await;

    assert_eq!(
        daemon_replies[0],
        DaemonResponse::ImageGenerationFailed {
            reason: ImageGenerationFailureReason::EngineBusy
        }
    );
    assert_eq!(
        stub_executor.logged_image_commands().len(),
        1,
        "the request was admitted first"
    );
}

#[tokio::test]
async fn should_reject_when_the_worker_serves_no_images() {
    let stub_executor = ready_stub_executor(CHAT_MODEL_ID);

    let (daemon_replies, _stub_executor) = serve_requests(
        "image-no-executor",
        stub_executor,
        vec![image_request(IMAGE_MODEL_ID, 512, 512, None)],
    )
    .await;

    assert!(rejection_reason_of(&daemon_replies[0]).contains("unavailable"));
}

#[tokio::test]
async fn should_list_which_installed_models_can_draw() {
    let stub_executor = ready_stub_executor(CHAT_MODEL_ID);

    let (daemon_replies, _stub_executor) = serve_requests(
        "image-models-list",
        stub_executor,
        vec![DaemonRequest::ModelsList],
    )
    .await;

    let DaemonResponse::ModelsList { models } = &daemon_replies[0] else {
        panic!("the daemon should answer with the model list");
    };
    let image_flags: Vec<(&str, bool)> = models.iter().map(listed_model_image_flag).collect();
    assert!(
        image_flags.contains(&(IMAGE_MODEL_ID, true)),
        "{image_flags:?}"
    );
    assert!(
        image_flags.contains(&(CHAT_MODEL_ID, false)),
        "{image_flags:?}"
    );
}

fn listed_model_image_flag(listed_model: &DaemonListedModel) -> (&str, bool) {
    (
        listed_model.model_id.as_str(),
        listed_model.supports_image_generation,
    )
}

/// The first command a new user runs must find its model: the built-in
/// default has to be an image-capable entry in the catalog this release ships,
/// or `astronomical image` would fail with "not in the release catalog" on a
/// fresh install. A catalog edit that drops or renames the model fails here.
#[test]
fn should_ship_the_builtin_default_image_model_in_the_release_catalog() {
    let release_catalog = astronomical_daemon::DownloadCatalog::load_bundled()
        .expect("the bundled release catalog should load");
    let default_model_id = astronomical_config::BUILTIN_DEFAULT_IMAGE_MODEL_ID;

    let mut default_catalog_entry = None;
    for catalog_entry in release_catalog.entries() {
        if astronomical_config::leaf_model_id(catalog_entry.huggingface_id()) == default_model_id {
            default_catalog_entry = Some(catalog_entry);
        }
    }

    let default_catalog_entry =
        default_catalog_entry.expect("the built-in image default must be in the release catalog");
    assert!(
        default_catalog_entry
            .capabilities()
            .supports_image_generation,
        "the built-in image default must be an image-generation model"
    );
}
