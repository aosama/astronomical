//! Bundled-catalog removal contract for the Qwen 3.8 35B-A3B Distill oQ6e MTP
//! 6-bit offer: the release catalog must no longer list this model. Installed
//! copies stay discoverable and serveable through disk discovery, which
//! resolves the installed artifact by leaf id and is independent of the
//! bundled catalog.

use astronomical_supervisor::DownloadCatalog;

#[test]
fn should_no_longer_offer_qwen_3_8_35b_a3b_distill_oq6e_from_the_bundled_catalog() {
    let download_catalog = DownloadCatalog::load_bundled()
        .expect("the bundled production catalog should remain valid");
    let oq6e_entry = download_catalog
        .entries()
        .iter()
        .find(|entry| entry.huggingface_id() == "funnygeeker/Qwen3.8-35B-A3B-Distill-oQ6e-mtp");
    assert!(
        oq6e_entry.is_none(),
        "the release catalog must no longer offer the Qwen 3.8 35B-A3B Distill oQ6e MTP 6-bit model"
    );
}
