//! The folded ternary family's catalog entries: the engine that serves them
//! hosts no drafter, so their text must not promise drafted speed.

use crate::DownloadCatalog;
use crate::download_catalog_tests::shipped_entry;

/// Words a catalog text uses to promise that a draft model makes replies
/// quicker.
const DRAFTING_SPEED_PROMISES: [&str; 4] = ["speeds it up", "accelerat", "quicker", "faster"];

/// The folded ternary engine hosts no drafter (issue #2595), so no entry it
/// serves may promise drafted speed, and the one that downloads a drafter
/// says plainly that the drafter goes unused.
#[test]
fn should_not_promise_drafted_speed_on_the_folded_ternary_family() {
    let download_catalog =
        DownloadCatalog::load_bundled().expect("the shipped bundled catalog should parse");
    let folded_ternary_entries: Vec<_> = download_catalog
        .entries()
        .iter()
        .filter(|catalog_entry| catalog_entry.family().as_str() == "prism_hadamard_qwen35")
        .collect();
    assert!(!folded_ternary_entries.is_empty());
    for folded_ternary_entry in folded_ternary_entries {
        let catalog_text = format!(
            "{} {}",
            folded_ternary_entry.summary().unwrap_or_default(),
            folded_ternary_entry.description().unwrap_or_default()
        )
        .to_lowercase();
        for speed_promise in DRAFTING_SPEED_PROMISES {
            assert!(
                !catalog_text.contains(speed_promise),
                "{} promises drafted speed (`{speed_promise}`): {catalog_text}",
                folded_ternary_entry.huggingface_id()
            );
        }
    }
    let drafter_bundle = shipped_entry(
        &download_catalog,
        "nathansutton/Qwen3.8-27B-Ternary-Bonsai-2-DFlash2-MLX",
    );
    for unused_drafter_text in [drafter_bundle.summary(), drafter_bundle.description()] {
        assert!(
            unused_drafter_text
                .unwrap_or_default()
                .contains("does not use yet"),
            "the drafter bundle must say its draft model is unused: {unused_drafter_text:?}"
        );
    }
}
