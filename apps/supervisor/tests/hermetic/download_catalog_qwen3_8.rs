//! Bundled-catalog contracts for the Qwen 3.8 catalog entries: the release
//! catalog must no longer list the Qwen 3.8 35B-A3B Distill oQ6e MTP 6-bit
//! offer, and it must offer the Qwen 3.8 27B MTPLX Optimized Speed artifact
//! that the `dense_mtp` acceptance role retargets to. Installed copies stay
//! discoverable and serveable through disk discovery, which resolves the
//! installed artifact by leaf id and is independent of the bundled catalog.

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

#[test]
fn should_offer_the_qwen_3_8_27b_mtplx_optimized_speed_artifact() {
    let download_catalog = DownloadCatalog::load_bundled()
        .expect("the bundled production catalog should remain valid");
    let mtplx_entry = download_catalog
        .entries()
        .iter()
        .find(|entry| entry.huggingface_id() == "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed")
        .expect("the release catalog should offer the Qwen 3.8 27B MTPLX Optimized Speed model");
    assert_eq!(
        mtplx_entry.revision(),
        "1d5087d2062c02b279180a53e4016cf9cd7a3d7e",
        "the MTPLX offer must stay pinned to its reviewed immutable revision"
    );
    assert!(
        mtplx_entry.capabilities().supports_vision,
        "the MTPLX artifact ships a vision sidecar and must advertise vision"
    );
}
