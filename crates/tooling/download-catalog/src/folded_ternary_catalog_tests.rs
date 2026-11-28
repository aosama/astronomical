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

/// Ternary Bonsai 2.5 supersedes Bonsai 2 at 27B, so the catalog offers the
/// 2.5 package alone rather than two 27B Bonsai builds side by side.
#[test]
fn should_offer_bonsai_2_5_as_the_only_27b_bonsai() {
    let download_catalog =
        DownloadCatalog::load_bundled().expect("the shipped bundled catalog should parse");
    let bonsai_27b_ids: Vec<&str> = download_catalog
        .entries()
        .iter()
        .map(crate::DownloadCatalogEntry::huggingface_id)
        .filter(|huggingface_id| {
            huggingface_id.contains("Bonsai") && huggingface_id.contains("27B")
        })
        .collect();
    assert_eq!(
        bonsai_27b_ids,
        ["ahmedihamdy/Ternary-Bonsai-2.5-27B-MLX-2bit-vision-DFLASH2"]
    );
}
