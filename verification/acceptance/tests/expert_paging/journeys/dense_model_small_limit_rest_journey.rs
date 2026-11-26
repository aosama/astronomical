//! A dense model under a limit far below its size (issues #1985 and #2221).
//!
//! A dense model has no experts to choose between: under a small limit the
//! engine keeps the leading decoder layers there is room for and reads the
//! others from the SSD on every pass. A person who gives a 9B model 3 GiB is
//! owed three things. The model is served and not refused. Memory in use
//! stays inside the limit while a long prompt is read. And once the long
//! prompt is over, as many layers are kept again as before it, so the next
//! request is not slower than the first.
//!
//! A limit just above the model's size is its own case. The whole model fits,
//! so it is loaded into memory, and then a prompt's MLX working buffers do not.
//! On 2026-11-14 a 9B model at 5 GiB answered every chat with an error
//! there, because only a model with experts was moved to the streaming
//! engine when a step did not fit. The second journey sends a short and a
//! long prompt at that limit and asserts both are served.

use std::net::SocketAddr;

use serde_json::{Value, json};
use serial_test::serial;
use tokio::time::{Duration, Instant, sleep, timeout};

use super::story_continuation_requests::{
    consume_completion, story_prompt, story_request, write_everyday_configuration,
};
use crate::support::{
    openai_client::LocalOpenAiClient,
    serving_rest::{get_json_endpoint, launch_real_model_rest_server, stop_real_model_rest_server},
};

const JOURNEY_LOG_LABEL: &str = "dense-small-limit";
const JOURNEY_BUDGET: Duration = Duration::from_secs(115);
/// The limit this acceptance cell runs under. Production code must not
/// assume it.
const MEMORY_LIMIT_BYTES: u64 = 3 * (1 << 30);
/// MLX may run one percent past the limit it is given.
const ALLOWED_ACTIVE_MEMORY_BYTES: u64 = MEMORY_LIMIT_BYTES + MEMORY_LIMIT_BYTES / 100;
/// A limit the model's weights fit under with almost nothing to spare. It
/// defines this acceptance cell only.
const BARELY_FITTING_MEMORY_LIMIT_BYTES: u64 = 5 * (1 << 30);
const SHORT_PROMPT_TOKENS: usize = 450;
const LONG_PROMPT_TOKENS: usize = 6_000;
const MAXIMUM_OUTPUT_TOKENS: u64 = 16;
const STATUS_POLL_INTERVAL: Duration = Duration::from_millis(100);
/// How long the engine may take to read layers back once it is idle.
const SETTLE_BUDGET: Duration = Duration::from_secs(20);

/// How many decoder layers the status page reports as kept, and of how many.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct KeptDenseLayers {
    kept_layer_count: u64,
    decoder_layer_count: u64,
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "loads a dense model under a 3 GiB limit and reads a long prompt after a short one"]
#[serial]
async fn should_serve_a_dense_model_inside_a_small_limit_and_keep_its_layers_afterwards() {
    timeout(JOURNEY_BUDGET, serve_a_dense_model_under_a_small_limit())
        .await
        .expect("the dense small-limit journey must finish inside the 120-second runner budget");
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "loads a dense model under a 5 GiB limit its weights barely fit and sends a short and a long prompt"]
#[serial]
async fn should_serve_a_dense_model_at_a_limit_its_weights_barely_fit() {
    timeout(
        JOURNEY_BUDGET,
        serve_a_dense_model_at_a_barely_fitting_limit(),
    )
    .await
    .expect("the dense barely-fitting journey must finish inside the 120-second runner budget");
}

async fn serve_a_dense_model_at_a_barely_fitting_limit() {
    let model_id = crate::support::dense_vision_model_id();
    let model_directory = crate::support::configured_installed_model_directory_by_id(model_id);
    let worker_home = tempfile::tempdir().expect("the isolated worker home should exist");
    write_everyday_configuration(worker_home.path());
    let prompts = [
        story_prompt(&model_directory, SHORT_PROMPT_TOKENS),
        story_prompt(&model_directory, LONG_PROMPT_TOKENS),
    ];
    let server = launch_real_model_rest_server(
        model_id,
        model_directory,
        worker_home.path(),
        BARELY_FITTING_MEMORY_LIMIT_BYTES,
    )
    .await;
    let client = LocalOpenAiClient::new(server.server_address, "dense-barely-fitting-limit");
    let mut served_prompt_count = 0_usize;
    for prompt in &prompts {
        let stream = client
            .create_streaming_chat_completion(&story_request(
                model_id,
                prompt,
                MAXIMUM_OUTPUT_TOKENS,
            ))
            .await
            .expect("the request should start");
        if consume_completion(stream, JOURNEY_LOG_LABEL).await {
            served_prompt_count += 1;
        }
    }
    let status_document = get_json_endpoint(server.server_address, "/v1/status").await;
    stop_real_model_rest_server(server).await;
    assert_eq!(
        served_prompt_count,
        prompts.len(),
        "a short and a long prompt must both be served at a limit the weights barely fit: {status_document}"
    );
}

async fn serve_a_dense_model_under_a_small_limit() {
    let model_id = crate::support::dense_vision_model_id();
    let model_directory = crate::support::configured_installed_model_directory_by_id(model_id);
    let worker_home = tempfile::tempdir().expect("the isolated worker home should exist");
    write_everyday_configuration(worker_home.path());
    let short_prompt = story_prompt(&model_directory, SHORT_PROMPT_TOKENS);
    let long_prompt = story_prompt(&model_directory, LONG_PROMPT_TOKENS);
    let server = launch_real_model_rest_server(
        model_id,
        model_directory,
        worker_home.path(),
        MEMORY_LIMIT_BYTES,
    )
    .await;
    let server_address = server.server_address;
    let client = LocalOpenAiClient::new(server_address, "dense-small-limit");

    // The model loads with the first request, so the first reading of its
    // kept layers is taken once that request is over.
    let short_stream = client
        .create_streaming_chat_completion(&story_request(
            model_id,
            &short_prompt,
            MAXIMUM_OUTPUT_TOKENS,
        ))
        .await
        .expect("the short request should start");
    assert!(
        consume_completion(short_stream, JOURNEY_LOG_LABEL).await,
        "the short request must be served"
    );
    let kept_layers_after_load = settled_kept_dense_layers(server_address, None).await;
    assert!(
        kept_layers_after_load.kept_layer_count > 0
            && kept_layers_after_load.kept_layer_count < kept_layers_after_load.decoder_layer_count,
        "a limit below the model's size must keep some layers and stream the others: {kept_layers_after_load:?}"
    );
    let long_stream = client
        .create_streaming_chat_completion(&story_request(
            model_id,
            &long_prompt,
            MAXIMUM_OUTPUT_TOKENS,
        ))
        .await
        .expect("the long request should start");
    let (long_request_was_served, largest_active_memory_bytes) = tokio::join!(
        consume_completion(long_stream, JOURNEY_LOG_LABEL),
        largest_active_memory_while_busy(server_address),
    );
    assert!(long_request_was_served, "the long request must be served");
    assert!(
        largest_active_memory_bytes <= ALLOWED_ACTIVE_MEMORY_BYTES,
        "memory in use must stay inside the limit while the long prompt is read: largest={largest_active_memory_bytes} allowed={ALLOWED_ACTIVE_MEMORY_BYTES}"
    );

    let kept_layers_after_long_prompt =
        settled_kept_dense_layers(server_address, Some(kept_layers_after_load)).await;
    stop_real_model_rest_server(server).await;
    eprintln!(
        "[dense-small-limit] kept_after_short_request={} kept_after_long_prompt={} of {} largest_active_gb={:.3}",
        kept_layers_after_load.kept_layer_count,
        kept_layers_after_long_prompt.kept_layer_count,
        kept_layers_after_load.decoder_layer_count,
        largest_active_memory_bytes as f64 / 1_073_741_824.0,
    );
    assert!(
        kept_layers_after_long_prompt.kept_layer_count >= kept_layers_after_load.kept_layer_count,
        "as many layers must be kept after the long prompt as before it: before={kept_layers_after_load:?} after={kept_layers_after_long_prompt:?}"
    );
}

/// Reads the kept-layer report once the engine is idle. With `at_least`,
/// waits up to the settle budget for that many layers to be kept again,
/// since layers are read back only after a request has ended.
async fn settled_kept_dense_layers(
    server_address: SocketAddr,
    at_least: Option<KeptDenseLayers>,
) -> KeptDenseLayers {
    let deadline = Instant::now() + SETTLE_BUDGET;
    loop {
        let status_document = get_json_endpoint(server_address, "/v1/status").await;
        let kept_dense_layers = kept_dense_layers_in(&status_document);
        let is_idle = status_document["activity"] == "idle";
        let has_settled = match (kept_dense_layers, at_least) {
            (Some(kept_now), Some(kept_before)) => {
                kept_now.kept_layer_count >= kept_before.kept_layer_count
            }
            (Some(_), None) => true,
            (None, _) => false,
        };
        if is_idle && (has_settled || Instant::now() >= deadline) {
            return kept_dense_layers.unwrap_or_else(|| {
                panic!("an idle dense model must report its kept layers: {status_document}")
            });
        }
        assert!(
            Instant::now() < deadline + SETTLE_BUDGET,
            "the engine did not become idle: {status_document}"
        );
        sleep(STATUS_POLL_INTERVAL).await;
    }
}

fn kept_dense_layers_in(status_document: &Value) -> Option<KeptDenseLayers> {
    let dense_layers = &status_document["expert_residency"]["dense_layers"];
    Some(KeptDenseLayers {
        kept_layer_count: dense_layers["kept_layer_count"].as_u64()?,
        decoder_layer_count: dense_layers["decoder_layer_count"].as_u64()?,
    })
}

/// The largest memory in use the status page showed from the moment the
/// engine became busy until it was idle again.
async fn largest_active_memory_while_busy(server_address: SocketAddr) -> u64 {
    let deadline = Instant::now() + JOURNEY_BUDGET;
    let mut largest_active_memory_bytes = 0_u64;
    let mut request_was_seen = false;
    while Instant::now() < deadline {
        let status_document = get_json_endpoint(server_address, "/v1/status").await;
        let is_idle = status_document["activity"] == "idle";
        request_was_seen |= !is_idle;
        if request_was_seen && is_idle {
            break;
        }
        largest_active_memory_bytes = largest_active_memory_bytes.max(
            status_document["mlx_memory_snapshot"]["active_memory_bytes"]
                .as_u64()
                .unwrap_or(0),
        );
        sleep(STATUS_POLL_INTERVAL).await;
    }
    largest_active_memory_bytes
}

#[test]
fn should_read_the_kept_layers_a_status_document_reports() {
    let status_document = json!({"expert_residency": {"dense_layers": {"kept_layer_count": 9, "decoder_layer_count": 32}}});
    assert_eq!(
        kept_dense_layers_in(&status_document),
        Some(KeptDenseLayers {
            kept_layer_count: 9,
            decoder_layer_count: 32,
        })
    );
}

#[test]
fn should_read_no_kept_layers_from_a_model_that_reports_none() {
    assert_eq!(kept_dense_layers_in(&json!({"expert_residency": {}})), None);
    assert_eq!(kept_dense_layers_in(&json!({})), None);
}

#[test]
fn should_read_no_kept_layers_when_a_count_is_not_a_number() {
    let status_document = json!({"expert_residency": {"dense_layers": {"kept_layer_count": "nine", "decoder_layer_count": 32}}});
    assert_eq!(kept_dense_layers_in(&status_document), None);
}
