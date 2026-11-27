//! The folded ternary family's catalog entries. Since #2595 the shared Qwen
//! 3.5 core serves these packs, with their vision tower and a bundled DFlash
//! drafter, so the entries must say what each download really does.

use crate::DownloadCatalog;
use crate::download_catalog_tests::shipped_entry;

const FOLDED_TERNARY_FAMILY: &str = "prism_hadamard_qwen35";

/// The packed drafter alone is about 1.2 GB, so a package that ships one
/// must declare more than its target's 8 GB of weights.
const FOLDED_TARGET_WEIGHTS_BYTES: u64 = 8_595_477_990;

#[test]
fn should_offer_the_bonsai_2_5_package_with_vision_and_its_drafter() {
    let download_catalog =
        DownloadCatalog::load_bundled().expect("the shipped bundled catalog should parse");
    let package_entry = shipped_entry(
        &download_catalog,
        "ahmedihamdy/Ternary-Bonsai-2.5-27B-MLX-2bit-vision-DFLASH2",
    );
    assert_eq!(package_entry.family().as_str(), FOLDED_TERNARY_FAMILY);
    assert_eq!(package_entry.revision().len(), 40);
    assert!(package_entry.capabilities().supports_vision);
    let selection = package_entry.download_path_selection();
    assert!(selection.includes("drafter/model.safetensors"));
    assert!(selection.includes("drafter/config.json"));
    assert!(selection.includes("hadamard.json"));
    assert!(package_entry.approximate_size_bytes() > FOLDED_TARGET_WEIGHTS_BYTES);
}

/// The community pack's drafter is served now, so its entry no longer says
/// it goes unused; it has no vision tower, so it still says text only.
#[test]
fn should_describe_the_community_bonsai_2_drafter_as_used_and_text_only() {
    let download_catalog =
        DownloadCatalog::load_bundled().expect("the shipped bundled catalog should parse");
    let drafter_bundle = shipped_entry(
        &download_catalog,
        "nathansutton/Qwen3.8-27B-Ternary-Bonsai-2-DFlash2-MLX",
    );
    let catalog_text = format!(
        "{} {}",
        drafter_bundle.summary().unwrap_or_default(),
        drafter_bundle.description().unwrap_or_default()
    );
    assert!(!catalog_text.contains("does not use"), "{catalog_text}");
    assert!(catalog_text.contains("Text only"), "{catalog_text}");
    assert!(!drafter_bundle.capabilities().supports_vision);
}

/// Prism ML's own Bonsai 2 pack carries a vision tower, which the shared core
/// serves, so its entry must not call it text only.
#[test]
fn should_offer_prism_mls_bonsai_2_pack_with_vision() {
    let download_catalog =
        DownloadCatalog::load_bundled().expect("the shipped bundled catalog should parse");
    let prism_pack = shipped_entry(&download_catalog, "prism-ml/Ternary-Bonsai-2-27B-mlx-2bit");
    assert!(prism_pack.capabilities().supports_vision);
    assert!(
        !prism_pack
            .summary()
            .unwrap_or_default()
            .contains("Text only")
    );
}
