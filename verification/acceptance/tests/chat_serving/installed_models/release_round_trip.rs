//! The release round trip: one installed chat model must load and stream a
//! healthy sampled completion, and one installed image model must load and
//! return a picture (issue #1411, narrowed by issue #2164).
//!
//! This is the release gate that loads real installed models through the
//! production worker and daemon: a loader change that refuses a real export
//! fails here at release time, not in front of a user after the release
//! ships. It deliberately loads two models, not the whole inventory. Loading
//! every installed model made a release's cost grow with whatever happened to
//! be on the developer's disk, while the family-specific journeys already
//! cover each family's loader; one chat model and one image model prove the
//! two serving paths a newcomer meets first.
//!
//! The smallest installed model of each kind is chosen, so the gate takes
//! seconds and its choice does not change between two runs on the same disk.
//! The chat model gets roughly one hundred tokens of Romeo and Juliet in at
//! temperature 1.0 — the path real replies take — and must stream a non-empty
//! reply within its one-hundred-token budget; an early end-of-sequence stop is
//! healthy behavior, so the assertion is on successful streaming and visible
//! text, never on an exact output count.
//!
//! The inventory is what the Development daemon itself serves: its own
//! Library models folder plus the configured model directories, discovered
//! with the daemon's own effective-discovery rules. Each round trip must
//! finish within its own 120-second budget.

use std::path::Path;
use std::time::Duration;

use astronomical_config::{
    AstronomicalConfig, AstronomicalInstancePaths, AstronomicalRuntimeInstance, DiscoveredModel,
    ModelCapabilities, discover_effective_models,
};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use image::ImageFormat;
use serde_json::json;
use tokio::time::timeout;

use crate::serving_acceptance::chat::openai_rest::{
    assert_successful_streaming_chat_response, post_chat_completion,
};
use crate::serving_acceptance::image_generation::flux2_klein_rest_support::{
    launch_flux_rest_server, post_image, response_json,
};
use crate::support::http::streamed_model_text_from_chat_response;
use crate::support::romeo_and_juliet;
use crate::support::serving_rest::launch_serving_rest_server_for_artifact_with_memory_limit;
use crate::support::serving_rest::stop_serving_rest_server;
/// Each model's load plus reply gets its own budget; every other journey in
/// this tree caps a single journey at this span.
const ROUND_TRIP_TIMEOUT_SECONDS: u64 = 120;
const MAXIMUM_OUTPUT_TOKENS: u16 = 100;
/// Roughly one hundred prompt tokens: eighty words of Romeo and Juliet with
/// the summary instruction.
const PROMPT_WORD_COUNT: usize = 80;
/// The smallest side every supported image family accepts: Qwen-Image
/// refuses anything under 256 pixels, and 256 is also a multiple of the
/// 16 and 32 pixel steps the families require. A small picture keeps the
/// gate to the model load plus a few denoising steps.
const PICTURE_SIDE_PIXELS: u32 = 256;
const PICTURE_SEED: u64 = 7_309;

#[tokio::test(flavor = "multi_thread")]
#[serial_test::serial]
#[ignore = "loads the smallest installed chat model and streams a sampled reply from it"]
async fn should_round_trip_one_installed_chat_model() {
    let development_models_directory = development_instance_paths().models_directory();
    let chat_model = smallest_installed_model(ModelKind::Chat).expect(
        "the release round trip needs one installed chat model; \
         install a chat model in Development or configure its model directories",
    );
    eprintln!(
        "[release-round-trip] status=start kind=chat model={}",
        chat_model.model_id
    );
    timeout(
        Duration::from_secs(ROUND_TRIP_TIMEOUT_SECONDS),
        round_trip_chat_model(&chat_model, &development_models_directory),
    )
    .await
    .unwrap_or_else(|_| {
        panic!(
            "model {} must load and reply within {ROUND_TRIP_TIMEOUT_SECONDS} seconds",
            chat_model.model_id
        )
    });
}

#[tokio::test(flavor = "multi_thread")]
#[serial_test::serial]
#[ignore = "loads the smallest installed image model and generates one small picture"]
async fn should_round_trip_one_installed_image_model() {
    let Some(image_model) = smallest_installed_model(ModelKind::ImageGeneration) else {
        // A Mac with no image model still releases: the chat round trip is
        // the required half, and the line below keeps the gap visible in the
        // release report.
        eprintln!("[release-round-trip] status=skipped kind=image reason=no-image-model-installed");
        return;
    };
    eprintln!(
        "[release-round-trip] status=start kind=image model={}",
        image_model.model_id
    );
    timeout(
        Duration::from_secs(ROUND_TRIP_TIMEOUT_SECONDS),
        round_trip_image_model(&image_model),
    )
    .await
    .unwrap_or_else(|_| {
        panic!(
            "model {} must load and draw within {ROUND_TRIP_TIMEOUT_SECONDS} seconds",
            image_model.model_id
        )
    });
}

#[test]
fn should_measure_a_model_folder_through_its_subfolders() {
    let model_folder = tempfile::tempdir().expect("a temporary model folder should be created");
    std::fs::write(model_folder.path().join("config.json"), [0_u8; 10])
        .expect("the top-level file should be written");
    let weights_folder = model_folder.path().join("transformer");
    std::fs::create_dir(&weights_folder).expect("the weights subfolder should be created");
    std::fs::write(weights_folder.join("weights.safetensors"), [0_u8; 32])
        .expect("the nested file should be written");

    assert_eq!(folder_size_bytes(model_folder.path()), 42);
}

#[test]
fn should_measure_an_empty_or_missing_model_folder_as_nothing() {
    let empty_folder = tempfile::tempdir().expect("a temporary folder should be created");

    assert_eq!(folder_size_bytes(empty_folder.path()), 0);
    assert_eq!(folder_size_bytes(&empty_folder.path().join("absent")), 0);
}

async fn round_trip_chat_model(chat_model: &DiscoveredModel, development_models_directory: &Path) {
    let model_id = chat_model.model_id.as_str();
    // The server runs in an isolated home with its own empty Library folder,
    // so Development's Library folder is listed there for Library-installed
    // models to be served.
    let isolated_development_home =
        crate::support::isolated_development_home_serving_models_from(development_models_directory);
    let rest_server = launch_serving_rest_server_for_artifact_with_memory_limit(
        chat_model.clone(),
        Some(isolated_development_home.path()),
        None,
        None,
    )
    .await;
    let chat_response = post_chat_completion(
        rest_server.server_address,
        json!({
            "model": model_id,
            "messages": [{
                "role": "user",
                "content": romeo_and_juliet::summary_prompt(PROMPT_WORD_COUNT),
            }],
            "stream": true,
            "temperature": 1.0,
            "max_tokens": MAXIMUM_OUTPUT_TOKENS,
        })
        .to_string(),
    )
    .await;
    assert_successful_streaming_chat_response(&chat_response);
    // Reasoning deltas count as streamed model output: a thinking model may
    // spend its whole budget reasoning, and that is healthy generation, not
    // a load failure.
    let streamed_text = streamed_model_text_from_chat_response(&chat_response);
    assert!(
        !streamed_text.trim().is_empty(),
        "model {model_id} must stream visible text at temperature 1.0 within the budget"
    );
    eprintln!(
        "[release-round-trip] status=success kind=chat model={model_id} reply_characters={}",
        streamed_text.chars().count()
    );
    stop_serving_rest_server(rest_server).await;
    drop(isolated_development_home);
}

async fn round_trip_image_model(image_model: &DiscoveredModel) {
    let model_id = image_model.model_id.as_str();
    // Only the helper's name is FLUX-specific: it launches the production
    // worker and daemon over Development's models, which serve every
    // installed image family.
    let rest_server = launch_flux_rest_server().await;
    let image_response = post_image(
        rest_server.server_address,
        json!({
            "model": model_id,
            "prompt": "A moonlit balcony in Verona.",
            "seed": PICTURE_SEED,
            "width": PICTURE_SIDE_PIXELS,
            "height": PICTURE_SIDE_PIXELS,
            "response_format": "b64_json",
        })
        .to_string(),
    )
    .await;
    assert!(
        image_response.starts_with("HTTP/1.1 200"),
        "model {model_id} must answer the picture request with success: {image_response:.400}"
    );
    let generated_picture = &response_json(&image_response)["data"][0];
    let png_bytes = STANDARD
        .decode(
            generated_picture["b64_json"]
                .as_str()
                .expect("the picture response should carry base64 PNG data"),
        )
        .expect("the picture data should be valid base64");
    let decoded_picture = image::load_from_memory_with_format(&png_bytes, ImageFormat::Png)
        .expect("the picture data should decode as a PNG");
    assert_eq!(
        (decoded_picture.width(), decoded_picture.height()),
        (PICTURE_SIDE_PIXELS, PICTURE_SIDE_PIXELS),
        "model {model_id} must return the picture size that was asked for"
    );
    eprintln!(
        "[release-round-trip] status=success kind=image model={model_id} png_bytes={}",
        png_bytes.len()
    );
    rest_server.stop().await;
}

/// The two kinds of model the release round trip loads.
#[derive(Clone, Copy)]
enum ModelKind {
    Chat,
    ImageGeneration,
}

impl ModelKind {
    fn matches(self, capabilities: &ModelCapabilities) -> bool {
        matches!(
            (self, capabilities),
            (Self::Chat, ModelCapabilities::Chat(_))
                | (Self::ImageGeneration, ModelCapabilities::ImageGeneration(_))
        )
    }
}

/// The smallest model of one kind that the Development daemon would serve.
/// Ties fall to the model id, so two runs on the same disk load the same one.
fn smallest_installed_model(model_kind: ModelKind) -> Option<DiscoveredModel> {
    let development_instance_paths = development_instance_paths();
    let astronomical_config =
        AstronomicalConfig::load_from_instance_paths(development_instance_paths.clone())
            .expect("the Development configuration should load for model discovery");
    let discovered_models = discover_effective_models(
        &development_instance_paths.models_directory(),
        astronomical_config.model_directories(),
    )
    .expect("model discovery should complete")
    .discovered_models;
    let mut smallest_model: Option<(u64, DiscoveredModel)> = None;
    for discovered_model in discovered_models {
        if !model_kind.matches(&discovered_model.capabilities) {
            continue;
        }
        let model_size_bytes = folder_size_bytes(&discovered_model.model_directory);
        let is_smaller = match &smallest_model {
            None => true,
            Some((smallest_size_bytes, smallest_so_far)) => {
                (model_size_bytes, &discovered_model.model_id)
                    < (*smallest_size_bytes, &smallest_so_far.model_id)
            }
        };
        if is_smaller {
            smallest_model = Some((model_size_bytes, discovered_model));
        }
    }
    let (_, smallest_discovered_model) = smallest_model?;
    Some(smallest_discovered_model)
}

/// The bytes a model folder holds, through its subfolders: image models keep
/// their weights one level down. A folder that cannot be read counts as
/// empty, since discovery has already validated the folders it returns.
fn folder_size_bytes(folder: &Path) -> u64 {
    let Ok(folder_entries) = std::fs::read_dir(folder) else {
        return 0;
    };
    let mut total_size_bytes = 0_u64;
    for folder_entry in folder_entries.flatten() {
        let Ok(entry_metadata) = std::fs::metadata(folder_entry.path()) else {
            continue;
        };
        total_size_bytes += if entry_metadata.is_dir() {
            folder_size_bytes(&folder_entry.path())
        } else {
            entry_metadata.len()
        };
    }
    total_size_bytes
}

fn development_instance_paths() -> AstronomicalInstancePaths {
    AstronomicalInstancePaths::default_location_instance_paths(
        AstronomicalRuntimeInstance::Development,
    )
    .expect("the Development instance paths should resolve for model discovery")
}
