//! Native official FLUX.2 Klein qualification through discovery and public REST.

use std::{net::SocketAddr, time::Duration};

use astronomical_config::{DiscoveredModel, ModelCapabilities, ModelFamily, ModelLicense};
use astronomical_model_serving::{
    FLUX2_KLEIN_OFFICIAL_MODEL_ID, FLUX2_KLEIN_OFFICIAL_REVISION, FLUX2_KLEIN_PROVIDER_MODEL_ID,
    Flux2KleinImageDimensions, flux2_klein_initial_latents_for_tests,
};
use astronomical_runtime_integration::{MlxDtype, MlxMemoryLimits, MlxRuntime};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use image::{GenericImageView, ImageFormat};
use serde_json::{Value, json};
use tokio::{
    task::JoinHandle,
    time::{sleep, timeout},
};

use super::{
    flux2_klein_rest_support::{
        FluxRestServer, assert_image_attribution, get_status, launch_flux_rest_server, post_image,
        response_json, wait_for_image_attribution_count,
    },
    model_artifact_rest_qualification::{
        assert_successful_streaming_chat_response, post_chat_completion,
    },
};
use crate::flux2_klein_reference_oracle::{ExpectedFluxReference, FluxReferenceOracle, sha256_hex};

const JOURNEY_TIMEOUT: Duration = Duration::from_secs(115);
const STATUS_POLL_INTERVAL: Duration = Duration::from_millis(100);
const QUALIFICATION_WIDTH_PIXELS: u32 = 64;
const QUALIFICATION_HEIGHT_PIXELS: u32 = 64;
const QUALIFICATION_SEED: u64 = 7_309;
const MAXIMUM_NATIVE_CHANNEL_DIFFERENCE: u8 = 1;
const ROMEO_AND_JULIET_SOURCE: &str =
    include_str!("../fixtures/model_metrics_5000_romeo_and_juliet_words.txt");

#[tokio::test(flavor = "multi_thread")]
#[ignore = "reads Development discovery to require the exact reviewed FLUX artifact"]
async fn should_discover_the_exact_pinned_flux2_klein_artifact() {
    eprintln!("[flux-qualification] phase=discovery status=started");
    timeout(JOURNEY_TIMEOUT, configured_official_flux_model())
        .await
        .expect("exact-artifact discovery must finish within 115 seconds");
    eprintln!("[flux-qualification] phase=discovery status=exact-pinned-artifact-found");
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires ASTRONOMICAL_FLUX2_KLEIN_REFERENCE_BUNDLE and the official artifact for native REST qualification"]
async fn should_qualify_official_flux2_klein_native_rest_generation() {
    timeout(JOURNEY_TIMEOUT, run_flux_rgb_and_repetition_journey())
        .await
        .expect("the official FLUX.2 Klein RGB journey must finish within 115 seconds");
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires ASTRONOMICAL_FLUX2_KLEIN_REFERENCE_BUNDLE for native initial-noise qualification"]
async fn should_match_the_independent_flux2_klein_initial_noise() {
    timeout(Duration::from_secs(15), run_initial_noise_journey())
        .await
        .expect("the native initial-noise journey must finish within 15 seconds");
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires the official artifact for cancellation cleanup and reuse qualification"]
async fn should_cleanup_and_reuse_flux2_klein_after_cancellation() {
    timeout(JOURNEY_TIMEOUT, run_flux_cancellation_reuse_journey())
        .await
        .expect("the FLUX.2 Klein cancellation journey must finish within 115 seconds");
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "swaps an exact configured chat fixture to official FLUX.2 Klein and back when available"]
async fn should_optionally_swap_from_chat_to_flux_and_back_to_chat() {
    timeout(JOURNEY_TIMEOUT, run_optional_chat_swap_journey())
        .await
        .expect("the optional chat-to-FLUX-to-chat journey must finish within 115 seconds");
}

async fn run_initial_noise_journey() {
    let qualification_prompt = qualification_prompt();
    let reference_oracle = read_reference_oracle(&qualification_prompt);
    assert_native_initial_noise_matches_reference(&reference_oracle).await;
}

async fn run_flux_rgb_and_repetition_journey() {
    let qualification_prompt = qualification_prompt();
    eprintln!(
        "[flux-qualification] phase=independent-reference prompt_sha256={}",
        sha256_hex(qualification_prompt.as_bytes())
    );
    let reference_oracle = read_reference_oracle(&qualification_prompt);
    let canonical_flux_model_id = astronomical_config::leaf_model_id(FLUX2_KLEIN_OFFICIAL_MODEL_ID);
    let rest_server = launch_flux_rest_server().await;
    let server_address = rest_server.server_address;
    let initial_status = get_status(server_address).await;
    let configuration_generation =
        initial_status["worker_runtime_feature_configuration"]["configuration_generation"]
            .as_str()
            .expect("the ready worker should acknowledge the resolved configuration generation")
            .to_owned();
    let request_body = flux_request_body(canonical_flux_model_id, &qualification_prompt);

    let first_response =
        post_image_with_progress(server_address, request_body.clone(), "first").await;
    if !first_response.starts_with("HTTP/1.1 200 OK") {
        eprintln!(
            "[flux-qualification] phase=diagnostics{}",
            rest_server.diagnostic_logs()
        );
    }
    let first_pixels = decode_qualified_rgb(&first_response);
    let reference_metrics = reference_oracle
        .compare_generated_rgb(&first_pixels)
        .unwrap_or_else(|comparison_error| panic!("{comparison_error}"));
    eprintln!(
        "[flux-qualification] phase=independent-reference status=matched max_channel_error={} mean_channel_error={:.6} p99_channel_error={} p999_channel_error={} channels_above_eight={}",
        reference_metrics.maximum_channel_error,
        reference_metrics.mean_channel_error,
        reference_metrics.p99_channel_error,
        reference_metrics.p999_channel_error,
        reference_metrics.channels_above_eight,
    );
    let status_after_first = wait_until_idle(server_address, "first-finalization").await;
    assert_worker_reuse_state(
        &status_after_first,
        canonical_flux_model_id,
        &configuration_generation,
    );

    let repeated_response = post_image_with_progress(server_address, request_body, "repeat").await;
    let repeated_pixels = decode_qualified_rgb(&repeated_response);
    let status_after_repeat = wait_until_idle(server_address, "repeat-finalization").await;
    assert_worker_reuse_state(
        &status_after_repeat,
        canonical_flux_model_id,
        &configuration_generation,
    );
    assert_native_repetition(&first_pixels, &repeated_pixels);
    assert_image_attribution(&rest_server, &["success", "success"]);
    rest_server.stop().await;
}

async fn run_flux_cancellation_reuse_journey() {
    let qualification_prompt = qualification_prompt();
    let canonical_flux_model_id = astronomical_config::leaf_model_id(FLUX2_KLEIN_OFFICIAL_MODEL_ID);
    let rest_server = launch_flux_rest_server().await;
    let server_address = rest_server.server_address;
    let initial_status = get_status(server_address).await;
    let configuration_generation =
        initial_status["worker_runtime_feature_configuration"]["configuration_generation"]
            .as_str()
            .expect("the ready worker should acknowledge the resolved configuration generation")
            .to_owned();
    let request_body = flux_request_body(canonical_flux_model_id, &qualification_prompt);

    cancel_image_after_progress(server_address, request_body.clone(), "cancel-first").await;
    wait_for_image_attribution_count(&rest_server, 1, "cancel-first-attribution").await;
    let status_after_first_cancel =
        wait_until_idle(server_address, "cancel-first-finalization").await;
    assert_worker_reuse_state(
        &status_after_first_cancel,
        canonical_flux_model_id,
        &configuration_generation,
    );

    let reuse_response =
        post_image_with_progress(server_address, request_body, "cancel-reuse").await;
    if !reuse_response.starts_with("HTTP/1.1 200 OK") {
        eprintln!(
            "[flux-qualification] phase=cancel-reuse-diagnostics{}",
            rest_server.diagnostic_logs()
        );
    }
    let _reuse_pixels = decode_qualified_rgb(&reuse_response);
    let status_after_reuse = wait_until_idle(server_address, "cancel-reuse-finalization").await;
    assert_worker_reuse_state(
        &status_after_reuse,
        canonical_flux_model_id,
        &configuration_generation,
    );
    wait_for_image_attribution_count(&rest_server, 2, "cancel-reuse-attribution").await;
    assert_image_attribution(&rest_server, &["cancelled", "success"]);
    rest_server.stop().await;
}

fn read_reference_oracle(qualification_prompt: &str) -> FluxReferenceOracle {
    let reference_oracle = FluxReferenceOracle::read_from_environment(&ExpectedFluxReference {
        model_id: FLUX2_KLEIN_PROVIDER_MODEL_ID,
        model_revision: FLUX2_KLEIN_OFFICIAL_REVISION,
        prompt: qualification_prompt,
        width: QUALIFICATION_WIDTH_PIXELS,
        height: QUALIFICATION_HEIGHT_PIXELS,
        seed: QUALIFICATION_SEED,
        steps: 4,
        guidance: 1.0,
    })
    .unwrap_or_else(|oracle_error| panic!("independent FLUX reference rejected: {oracle_error}"));
    eprintln!(
        "[flux-qualification] phase=independent-reference status=validated bfl_revision={} diffusers_revision={}",
        reference_oracle.bfl_source_revision(),
        reference_oracle.diffusers_source_revision()
    );
    reference_oracle
}

async fn assert_native_initial_noise_matches_reference(reference_oracle: &FluxReferenceOracle) {
    let initial_noise_sha256 = timeout(
        Duration::from_secs(10),
        tokio::task::spawn_blocking(native_initial_noise_sha256),
    )
    .await
    .expect("native initial-noise qualification must finish within 10 seconds")
    .expect("the native initial-noise qualification task must not panic");
    reference_oracle
        .verify_initial_noise_sha256(&initial_noise_sha256)
        .unwrap_or_else(|noise_error| panic!("independent FLUX noise rejected: {noise_error}"));
    eprintln!(
        "[flux-qualification] phase=independent-reference status=initial-noise-matched sha256={initial_noise_sha256}"
    );
}

fn native_initial_noise_sha256() -> String {
    let runtime = MlxRuntime::initialize(
        MlxMemoryLimits::new(512_000_000, 8_000_000)
            .expect("the qualification MLX memory limits should be valid"),
    )
    .expect("the pinned MLX runtime should initialize for noise qualification");
    let dimensions = Flux2KleinImageDimensions::validate(
        QUALIFICATION_WIDTH_PIXELS,
        QUALIFICATION_HEIGHT_PIXELS,
        1_000_000,
    )
    .expect("the qualification image dimensions should satisfy the production contract");
    let initial_latents =
        flux2_klein_initial_latents_for_tests(&runtime, QUALIFICATION_SEED, &dimensions)
            .expect("the production keyed-noise path should evaluate");
    let initial_latent_values = runtime
        .astype(&initial_latents, MlxDtype::Float32)
        .expect("the keyed BF16 noise should cast to float32 for its portable digest")
        .to_vec_f32()
        .expect("the keyed noise should materialize for qualification");
    let mut initial_latent_bytes = Vec::with_capacity(initial_latent_values.len() * 4);
    for initial_latent_value in initial_latent_values {
        initial_latent_bytes.extend_from_slice(&initial_latent_value.to_le_bytes());
    }
    sha256_hex(&initial_latent_bytes)
}

async fn run_optional_chat_swap_journey() {
    let discovered_models = configured_discovered_models().await;
    let flux_model = official_flux_model_from_discovery(&discovered_models);
    let Some(chat_model) = discovered_models.into_iter().find(|model| {
        model.model_id == crate::common::ORNITH_MODEL_ARTIFACT_QUALIFICATION_MODEL_ID
            && matches!(&model.capabilities, ModelCapabilities::Chat(_))
    }) else {
        eprintln!("[flux-swap] phase=skipped reason=exact-chat-fixture-not-configured");
        return;
    };
    let rest_server = launch_flux_rest_server().await;
    send_chat_litmus(&rest_server, &chat_model.model_id, "chat-before-flux").await;
    let image_response = post_image_with_progress(
        rest_server.server_address,
        flux_request_body(&flux_model.model_id, &qualification_prompt()),
        "swap-flux",
    )
    .await;
    let _generated_pixels = decode_qualified_rgb(&image_response);
    send_chat_litmus(&rest_server, &chat_model.model_id, "chat-after-flux").await;
    let final_status = wait_until_idle(rest_server.server_address, "chat-swap-final").await;
    assert_eq!(final_status["ready_model_id"], chat_model.model_id);
    rest_server.stop().await;
}

async fn configured_official_flux_model() -> DiscoveredModel {
    official_flux_model_from_discovery(&configured_discovered_models().await)
}

async fn configured_discovered_models() -> Vec<DiscoveredModel> {
    timeout(
        JOURNEY_TIMEOUT,
        tokio::task::spawn_blocking(crate::common::configured_discovered_models),
    )
    .await
    .expect("Development model discovery must finish within 115 seconds")
    .expect("the Development model-discovery task must not panic")
}

fn official_flux_model_from_discovery(discovered_models: &[DiscoveredModel]) -> DiscoveredModel {
    let canonical_model_id = astronomical_config::leaf_model_id(FLUX2_KLEIN_OFFICIAL_MODEL_ID);
    let discovered_model = discovered_models
        .iter()
        .find(|model| {
            model.model_family == ModelFamily::Flux2Klein
                && model.model_id == canonical_model_id
        })
        .cloned()
        .expect(
            "Development model_directories should discover the exact reviewed FLUX.2 Klein artifact",
        );
    assert_eq!(
        discovered_model.provider_model_id.as_deref(),
        Some(FLUX2_KLEIN_PROVIDER_MODEL_ID)
    );
    assert_eq!(discovered_model.revision, FLUX2_KLEIN_OFFICIAL_REVISION);
    assert_eq!(discovered_model.license, Some(ModelLicense::Apache20));
    assert!(matches!(
        &discovered_model.capabilities,
        ModelCapabilities::ImageGeneration(capabilities)
            if capabilities.supports_text_to_image
                && !capabilities.supports_image_editing
                && !capabilities.supports_multiple_reference_images
    ));
    discovered_model
}

fn qualification_prompt() -> String {
    let source_excerpt = ROMEO_AND_JULIET_SOURCE
        .chars()
        .take(240)
        .collect::<String>();
    format!("A moonlit balcony scene inspired by this Romeo and Juliet excerpt: {source_excerpt}")
}

fn flux_request_body(canonical_model_id: &str, qualification_prompt: &str) -> String {
    json!({
        "model": canonical_model_id,
        "prompt": qualification_prompt,
        "seed": QUALIFICATION_SEED,
        "width": QUALIFICATION_WIDTH_PIXELS,
        "height": QUALIFICATION_HEIGHT_PIXELS,
        "steps": 4,
        "guidance": 1.0,
        "response_format": "b64_json"
    })
    .to_string()
}

async fn post_image_with_progress(
    server_address: SocketAddr,
    request_body: String,
    request_label: &'static str,
) -> String {
    let image_task = tokio::spawn(post_image(server_address, request_body));
    report_progress_until_finished(server_address, request_label, &image_task).await;
    image_task
        .await
        .expect("the image HTTP task should not panic")
}

async fn report_progress_until_finished(
    server_address: SocketAddr,
    request_label: &str,
    image_task: &JoinHandle<String>,
) {
    let mut last_progress = None;
    let mut poll_count = 0_u32;
    while !image_task.is_finished() {
        poll_count = poll_count.saturating_add(1);
        let status = get_status(server_address).await;
        let progress = (
            status["progress"]["phase"]
                .as_str()
                .unwrap_or("model_loading")
                .to_owned(),
            status["progress"]["completed_steps"].as_u64().unwrap_or(0),
            status["progress"]["total_steps"].as_u64().unwrap_or(4),
        );
        if last_progress.as_ref() != Some(&progress) || poll_count.is_multiple_of(50) {
            eprintln!(
                "[flux-qualification] request={request_label} phase={} step={}/{}",
                progress.0, progress.1, progress.2
            );
            last_progress = Some(progress);
        }
        sleep(STATUS_POLL_INTERVAL).await;
    }
}

async fn cancel_image_after_progress(
    server_address: SocketAddr,
    request_body: String,
    request_label: &'static str,
) {
    let image_task = tokio::spawn(post_image(server_address, request_body));
    let mut poll_count = 0_u32;
    loop {
        poll_count = poll_count.saturating_add(1);
        let status = get_status(server_address).await;
        let progress_phase = status["progress"]["phase"].as_str();
        if status["activity"] == "image_generation"
            && progress_phase.is_some_and(|phase| phase != "model_loading")
        {
            eprintln!(
                "[flux-qualification] request={request_label} phase={} action=disconnect",
                progress_phase.expect("the image execution phase should be present")
            );
            break;
        }
        if image_task.is_finished() {
            let completed_response = image_task
                .await
                .expect("the completed cancellation HTTP task should not panic");
            panic!("the cancellation request completed before disconnect: {completed_response}");
        }
        if poll_count.is_multiple_of(50) {
            eprintln!(
                "[flux-qualification] request={request_label} phase=waiting polls={poll_count}"
            );
        }
        sleep(STATUS_POLL_INTERVAL).await;
    }
    image_task.abort();
    let _cancelled_task_outcome = image_task.await;
}

async fn wait_until_idle(server_address: SocketAddr, phase_label: &str) -> Value {
    let mut poll_count = 0_u32;
    loop {
        poll_count = poll_count.saturating_add(1);
        let status = get_status(server_address).await;
        if status["status"] == "ready" && status["activity"] == "idle" {
            eprintln!("[flux-qualification] phase={phase_label} status=idle polls={poll_count}");
            return status;
        }
        if poll_count.is_multiple_of(10) {
            eprintln!("[flux-qualification] phase={phase_label} status=waiting polls={poll_count}");
        }
        sleep(STATUS_POLL_INTERVAL).await;
    }
}

fn decode_qualified_rgb(response: &str) -> Vec<u8> {
    assert!(
        response.starts_with("HTTP/1.1 200 OK"),
        "unexpected image response: {response}"
    );
    let response_document = response_json(response);
    let generated_image = &response_document["data"][0];
    assert_eq!(generated_image["mime_type"], "image/png");
    assert_eq!(
        generated_image["model_revision"],
        FLUX2_KLEIN_OFFICIAL_REVISION
    );
    assert_eq!(generated_image["seed"], QUALIFICATION_SEED);
    assert_eq!(generated_image["width"], QUALIFICATION_WIDTH_PIXELS);
    assert_eq!(generated_image["height"], QUALIFICATION_HEIGHT_PIXELS);
    let png_bytes = STANDARD
        .decode(
            generated_image["b64_json"]
                .as_str()
                .expect("the response should contain base64 PNG data"),
        )
        .expect("the generated image should contain valid base64");
    let decoded_image = image::load_from_memory_with_format(&png_bytes, ImageFormat::Png)
        .expect("the generated bytes should decode as PNG");
    assert_eq!(
        decoded_image.dimensions(),
        (QUALIFICATION_WIDTH_PIXELS, QUALIFICATION_HEIGHT_PIXELS)
    );
    decoded_image.into_rgb8().into_raw()
}

fn assert_native_repetition(first_pixels: &[u8], repeated_pixels: &[u8]) {
    assert_eq!(first_pixels.len(), repeated_pixels.len());
    // Seeded native graphs are stable while BF16 GPU reductions may cross one adjacent u8 boundary.
    // This pinned-runtime bound is deliberately not a cross-runtime numerical-parity claim.
    let maximum_difference = first_pixels
        .iter()
        .zip(repeated_pixels)
        .map(|(first_channel, repeated_channel)| first_channel.abs_diff(*repeated_channel))
        .max()
        .unwrap_or(0);
    assert!(
        maximum_difference <= MAXIMUM_NATIVE_CHANNEL_DIFFERENCE,
        "fixed native FLUX input exceeded the pinned-runtime RGB tolerance: {maximum_difference}"
    );
    eprintln!(
        "[flux-qualification] phase=native-repetition max_channel_difference={maximum_difference}"
    );
}

fn assert_worker_reuse_state(
    status: &Value,
    canonical_flux_model_id: &str,
    configuration_generation: &str,
) {
    assert_eq!(status["ready_model_id"], canonical_flux_model_id);
    assert_eq!(
        status["worker_runtime_feature_configuration"]["configuration_generation"].as_str(),
        Some(configuration_generation)
    );
    let loaded_model = &status["worker_runtime_feature_configuration"]["loaded_model"];
    assert_eq!(loaded_model["kind"], "flux2_klein");
    assert_eq!(
        loaded_model["configuration"]["model_id"],
        canonical_flux_model_id
    );
    assert_eq!(
        loaded_model["configuration"]["artifact_revision"],
        FLUX2_KLEIN_OFFICIAL_REVISION
    );
    let memory_snapshot = &status["mlx_memory_snapshot"];
    let memory_snapshot_source = memory_snapshot["source"].as_str();
    assert!(
        matches!(memory_snapshot_source, Some("finalized" | "idle_poll")),
        "status should retain finalized cleanup or a newer idle sample: {memory_snapshot}"
    );
    assert_eq!(
        required_u64(memory_snapshot, "allocator_cache_memory_bytes"),
        0
    );
    assert_eq!(required_u64(memory_snapshot, "expert_payload_bytes"), 0);
    assert_eq!(
        required_u64(memory_snapshot, "context_state_payload_bytes"),
        0
    );
    assert_eq!(
        required_u64(memory_snapshot, "speculative_prefill_draft_memory_bytes"),
        0
    );
    let active_memory_bytes = required_u64(memory_snapshot, "active_memory_bytes");
    let peak_memory_bytes = required_u64(memory_snapshot, "peak_memory_bytes");
    let mlx_memory_ceiling_bytes = required_u64(status, "mlx_memory_ceiling_bytes");
    assert!(active_memory_bytes <= peak_memory_bytes);
    assert!(active_memory_bytes <= mlx_memory_ceiling_bytes);
}

fn required_u64(document: &Value, field_name: &str) -> u64 {
    document[field_name]
        .as_u64()
        .unwrap_or_else(|| panic!("{field_name} must contain numeric memory telemetry: {document}"))
}

async fn send_chat_litmus(rest_server: &FluxRestServer, model_id: &str, phase: &str) {
    eprintln!("[flux-swap] phase={phase} model={model_id}");
    let source_excerpt = ROMEO_AND_JULIET_SOURCE
        .chars()
        .take(400)
        .collect::<String>();
    let response = post_chat_completion(
        rest_server.server_address,
        json!({
            "model": model_id,
            "messages": [{"role": "user", "content": format!(
                "Use this Romeo and Juliet excerpt and name one household: {source_excerpt}"
            )}],
            "stream": true,
            "temperature": 1,
            "max_tokens": 8
        })
        .to_string(),
    )
    .await;
    if !response.starts_with("HTTP/1.1 200 OK") {
        eprintln!(
            "[flux-swap] phase={phase}-diagnostics{}",
            rest_server.diagnostic_logs()
        );
    }
    assert_successful_streaming_chat_response(&response);
}
