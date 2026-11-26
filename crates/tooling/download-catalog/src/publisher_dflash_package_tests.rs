//! The Ornith team's own first-generation DFlash drafter, packed in a `drafter`
//! folder beside their MLX builds with the vision tower put back, and beside the
//! OptiQ build, whose tower lives in `optiq/`. The download must keep the
//! drafter and the tower, and the size the disk check reserves must count the
//! drafter.

use crate::DownloadCatalog;
use crate::download_catalog_tests::shipped_entry;

const PUBLISHER_DRAFTER_WEIGHTS_BYTES: u64 = 771_819_680;

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

#[test]
fn should_ship_the_optiq_vision_dflash_package_as_a_drafting_build_of_the_optiq_lead() {
    let download_catalog =
        DownloadCatalog::load_bundled().expect("the shipped bundled catalog should parse");
    let package_entry = shipped_entry(
        &download_catalog,
        "ahmedihamdy/Ornith-1.5-35B-A3B-OptiQ-4bit-vision-DFLASH",
    );
    let lead_entry = shipped_entry(
        &download_catalog,
        "mlx-community/Ornith-1.5-35B-A3B-OptiQ-4bit",
    );
    assert_eq!(package_entry.family().as_str(), "qwen3_5");
    assert_eq!(package_entry.revision().len(), 40);
    assert!(!package_entry.experimental() && !package_entry.get_started());
    assert!(package_entry.capabilities().supports_vision);
    let selection = package_entry.download_path_selection();
    assert!(selection.includes("drafter/model.safetensors"));
    assert!(selection.includes("drafter/config.json"));
    assert!(selection.includes("optiq/optiq_vision.safetensors"));
    // The multi-token-prediction head is left out, so the package is smaller
    // than its lead by that head, minus the drafter it carries instead.
    assert!(package_entry.approximate_size_bytes() > PUBLISHER_DRAFTER_WEIGHTS_BYTES);
    assert!(
        package_entry.approximate_size_bytes()
            < lead_entry.approximate_size_bytes() + PUBLISHER_DRAFTER_WEIGHTS_BYTES
    );
    assert!(package_entry.approximate_memory_bytes() > package_entry.approximate_size_bytes());
}

#[test]
fn should_record_the_optiq_vision_dflash_package_as_faster_than_its_lead_at_every_task() {
    let download_catalog =
        DownloadCatalog::load_bundled().expect("the shipped bundled catalog should parse");
    let speed_of = |huggingface_id| {
        shipped_entry(&download_catalog, huggingface_id)
            .measured_speed()
            .expect("both builds were measured")
            .tokens_per_second()
    };
    let package_speed = speed_of("ahmedihamdy/Ornith-1.5-35B-A3B-OptiQ-4bit-vision-DFLASH");
    let lead_speed = speed_of("mlx-community/Ornith-1.5-35B-A3B-OptiQ-4bit");
    assert!(package_speed.code_short_prompt > lead_speed.code_short_prompt);
    assert!(package_speed.code_long_prompt > lead_speed.code_long_prompt);
    assert!(package_speed.essay_short_prompt > lead_speed.essay_short_prompt);
}
