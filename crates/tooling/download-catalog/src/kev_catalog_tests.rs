//! Kev's catalog entry: the published merged MLX build of Kev-0.8B, which
//! the Kev engine reads as it is (`decision.json` beside `head.safetensors`).

use crate::download_catalog_tests::shipped_entry;
use crate::{DownloadCatalog, DownloadCatalogRole};

const KEV_HUGGINGFACE_ID: &str = "AbstractFramework/kev-0.8b-mlx-8bit";

/// The entry downloads the backbone, the float32 pointer head and the
/// settings that mark the folder as Kev, without the build's README or the
/// tokenizer files the engine never reads.
#[test]
fn should_ship_kev_as_a_decision_model_with_its_head_settings_and_limits() {
    let download_catalog =
        DownloadCatalog::load_bundled().expect("the shipped bundled catalog should parse");
    let kev_entry = shipped_entry(&download_catalog, KEV_HUGGINGFACE_ID);

    assert_eq!(kev_entry.role(), DownloadCatalogRole::Decision);
    assert_eq!(kev_entry.family().as_str(), "kev");
    assert!(kev_entry.capabilities().supports_decisions);
    assert_eq!(kev_entry.upstream_license(), Some("Apache-2.0"));
    let selection = kev_entry.download_path_selection();
    assert!(selection.includes("decision.json"));
    assert!(selection.includes("head.safetensors"));
    assert!(selection.includes("model.safetensors"));
    assert!(selection.includes("tokenizer.json"));
    assert!(!selection.includes("README.md"));
    assert!(!selection.includes("vocab.json"));
}

/// The description states the limits a person meets: the state's token
/// budget, and that a longer text is refused rather than cut.
#[test]
fn should_state_kevs_token_limit_and_that_a_longer_text_is_refused() {
    let download_catalog =
        DownloadCatalog::load_bundled().expect("the shipped bundled catalog should parse");
    let description = shipped_entry(&download_catalog, KEV_HUGGINGFACE_ID)
        .description()
        .expect("the Kev entry should state its limits");

    assert!(description.contains("65,536 tokens"));
    assert!(description.contains("refused, not cut"));
}

/// The memory figure is the peak `cargo measure-kev-decision-latency`
/// printed at the 65,536-token state limit, not an estimate from the weight
/// files, and it is more than the files, which a forward pass works beside.
#[test]
fn should_state_kev_memory_as_the_peak_measured_at_its_token_limit() {
    let download_catalog =
        DownloadCatalog::load_bundled().expect("the shipped bundled catalog should parse");
    let kev_entry = shipped_entry(&download_catalog, KEV_HUGGINGFACE_ID);

    assert_eq!(kev_entry.approximate_memory_bytes(), 2_915_511_786);
    assert!(kev_entry.approximate_memory_bytes() > kev_entry.approximate_size_bytes());
}
