//! The release variant round trip: every installed chat-capable model variant
//! must load and stream a healthy sampled completion (issue #1411).
//!
//! This is the gate that catches pack variants the single-pack journeys
//! structurally miss: a real installed export whose config shape the loader
//! refuses fails here at release time, not in front of a user after the
//! release ships. Each discovered chat model gets roughly one hundred tokens
//! of Romeo and Juliet in at temperature 1.0 — the path real replies take —
//! and must stream a non-empty reply within its one-hundred-token budget; an
//! early end-of-sequence stop is healthy behavior, not a defect, so the
//! assertion is on successful streaming and visible text, never on an exact
//! output count.
//!
//! The inventory comes from the machine's configured model roots, or from
//! `ASTRONOMICAL_RELEASE_VARIANT_MODEL_ROOTS` (colon-separated) when a
//! release run wants an explicit variant set. Each model round trip must
//! finish within its own 120-second budget.

use std::path::PathBuf;
use std::time::Duration;

use astronomical_config::{AstronomicalConfig, ModelCapabilities};
use serde_json::json;
use tokio::time::timeout;

use crate::serving_acceptance::chat::openai_rest::{
    assert_successful_streaming_chat_response, post_chat_completion,
};
use crate::support::http::streamed_model_text_from_chat_response;
use crate::support::romeo_and_juliet;
use crate::support::serving_rest::launch_serving_rest_server_for_artifact_with_memory_limit;
use crate::support::serving_rest::stop_serving_rest_server;
/// Each model's load plus completion gets its own budget; every other
/// journey in this tree caps a single journey at this span.
const VARIANT_ROUND_TRIP_TIMEOUT_SECONDS: u64 = 120;
const MAXIMUM_OUTPUT_TOKENS: u16 = 100;
/// Roughly one hundred prompt tokens: eighty words of Romeo and Juliet with
/// the summary instruction.
const PROMPT_WORD_COUNT: usize = 80;
const VARIANT_MODEL_ROOTS_ENV: &str = "ASTRONOMICAL_RELEASE_VARIANT_MODEL_ROOTS";

#[tokio::test(flavor = "multi_thread")]
#[ignore = "loads every installed chat-capable model variant and streams a sampled reply from each"]
async fn should_round_trip_every_installed_chat_model_variant() {
    let chat_model_variants = discovered_chat_model_variants();
    assert!(
        !chat_model_variants.is_empty(),
        "the release variant round trip needs a populated chat-model inventory; \
         set {VARIANT_MODEL_ROOTS_ENV} or configure model directories"
    );
    for discovered_model in &chat_model_variants {
        eprintln!(
            "[variant-round-trip] status=start model={}",
            discovered_model.model_id
        );
        timeout(
            Duration::from_secs(VARIANT_ROUND_TRIP_TIMEOUT_SECONDS),
            round_trip_one_model_variant(discovered_model),
        )
        .await
        .unwrap_or_else(|_| {
            panic!(
                "model {} must load and reply within {VARIANT_ROUND_TRIP_TIMEOUT_SECONDS} seconds",
                discovered_model.model_id
            )
        });
    }
    eprintln!(
        "[variant-round-trip] status=success variant_count={}",
        chat_model_variants.len()
    );
}

async fn round_trip_one_model_variant(discovered_model: &astronomical_config::DiscoveredModel) {
    let model_id = discovered_model.model_id.as_str();
    let isolated_development_home = crate::support::isolated_development_home_from_user_config();
    let rest_server = launch_serving_rest_server_for_artifact_with_memory_limit(
        discovered_model.clone(),
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
        "[variant-round-trip] status=success model={model_id} reply_characters={}",
        streamed_text.chars().count()
    );
    stop_serving_rest_server(rest_server).await;
    drop(isolated_development_home);
}

/// Every chat-capable model the release machine's roots discover, in a
/// stable order so consecutive release runs report comparably.
fn discovered_chat_model_variants() -> Vec<astronomical_config::DiscoveredModel> {
    let astronomical_config = AstronomicalConfig::load_from_development_location()
        .expect("the Development configuration should load for variant discovery");
    let model_roots: Vec<PathBuf> = match std::env::var(VARIANT_MODEL_ROOTS_ENV) {
        Ok(explicit_roots) if !explicit_roots.is_empty() => explicit_roots
            .split(':')
            .filter(|root| !root.is_empty())
            .map(PathBuf::from)
            .collect(),
        _ => astronomical_config.model_directories().to_vec(),
    };
    let mut chat_model_variants: Vec<astronomical_config::DiscoveredModel> =
        astronomical_config::discover_models(&model_roots)
            .expect("variant discovery should complete")
            .into_iter()
            .flat_map(|configured_model_directory_scan| {
                configured_model_directory_scan.discovered_models
            })
            .filter(|discovered_model| {
                matches!(discovered_model.capabilities, ModelCapabilities::Chat(_))
            })
            .collect();
    chat_model_variants.sort_by(|first, second| first.model_id.cmp(&second.model_id));
    chat_model_variants
}
