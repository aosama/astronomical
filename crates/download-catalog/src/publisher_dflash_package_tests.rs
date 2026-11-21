//! The Ornith team's own MLX builds with their vision tower put back and their
//! own first-generation DFlash drafter in a `drafter` folder. The download must
//! keep the drafter and the tower, and the size the disk check reserves must
//! count the drafter.

use crate::{DownloadCatalog, DownloadCatalogEntry};

const PUBLISHER_DRAFTER_WEIGHTS_BYTES: u64 = 771_819_680;

fn shipped_entry<'catalog>(
    download_catalog: &'catalog DownloadCatalog,
    huggingface_id: &str,
) -> &'catalog DownloadCatalogEntry {
    download_catalog
        .entries()
        .iter()
        .find(|catalog_entry| catalog_entry.huggingface_id() == huggingface_id)
        .unwrap_or_else(|| panic!("the shipped catalog should offer {huggingface_id}"))
}

#[test]
fn should_ship_the_publisher_vision_dflash_packages_with_their_drafter_and_tower() {
    let download_catalog =
        DownloadCatalog::load_bundled().expect("the shipped bundled catalog should parse");
    for package_huggingface_id in [
        "ahmedihamdy/Ornith-1.5-35B-A3B-MLX-4bit-vision-DFLASH",
        "ahmedihamdy/Ornith-1.5-35B-A3B-MLX-6bit-vision-DFLASH",
    ] {
        let package_entry = shipped_entry(&download_catalog, package_huggingface_id);
        assert_eq!(package_entry.family().as_str(), "qwen3_5");
        assert_eq!(package_entry.revision().len(), 40);
        assert!(!package_entry.experimental() && !package_entry.get_started());
        assert!(package_entry.capabilities().supports_vision);
        let selection = package_entry.download_path_selection();
        assert!(selection.includes("drafter/model.safetensors"));
        assert!(selection.includes("drafter/config.json"));
        assert!(selection.includes("model-vision.safetensors"));
        assert!(package_entry.approximate_size_bytes() > PUBLISHER_DRAFTER_WEIGHTS_BYTES);
        assert!(package_entry.approximate_memory_bytes() > package_entry.approximate_size_bytes());
        assert!(package_entry.minimum_memory_bytes().is_some());
    }
}

#[test]
fn should_not_offer_the_text_only_publisher_dflash_packages() {
    let download_catalog =
        DownloadCatalog::load_bundled().expect("the shipped bundled catalog should parse");
    for text_only_huggingface_id in [
        "ahmedihamdy/Ornith-1.5-35B-A3B-MLX-4bit-DFLASH",
        "ahmedihamdy/Ornith-1.5-35B-A3B-MLX-6bit-DFLASH",
    ] {
        assert!(
            download_catalog
                .entries()
                .iter()
                .all(|catalog_entry| catalog_entry.huggingface_id() != text_only_huggingface_id),
            "{text_only_huggingface_id} has no vision tower and is not offered"
        );
    }
}
