//! Bundled-catalog contract for the Qwen 3.6 35B-A3B OptiQ 4-bit offer.

use astronomical_supervisor::{DownloadCatalog, DownloadCatalogFamily};

#[test]
fn should_offer_qwen_3_6_35b_a3b_optiq_4bit_from_the_bundled_catalog() {
    let download_catalog = DownloadCatalog::load_bundled()
        .expect("the bundled production catalog should remain valid");
    let qwen_entry = download_catalog
        .entries()
        .iter()
        .find(|entry| entry.huggingface_id() == "mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit")
        .expect("the release catalog should include the Qwen 3.6 35B-A3B OptiQ 4-bit offer");

    // The pinned revision is the contract: it must match the reviewed Hub tree,
    // not whatever the repository head happens to be.
    assert_eq!(
        qwen_entry.revision(),
        "70a3aa32c7feef511182bf16aa332f37e8d82014"
    );
    assert_eq!(qwen_entry.family(), DownloadCatalogFamily::Qwen3_5);
    assert_eq!(qwen_entry.approximate_size_bytes(), 24_693_956_069);
    assert_eq!(qwen_entry.upstream_license(), Some("Apache-2.0"));
    assert_eq!(
        qwen_entry.quantization_label(),
        Some("OptiQ mixed-precision 4/8-bit (affine, group 64)")
    );

    let capabilities = qwen_entry.capabilities();
    assert!(capabilities.supports_reasoning);
    assert!(capabilities.supports_vision);
    assert!(capabilities.supports_tool_calls);
    assert!(!capabilities.supports_embeddings);
    assert!(!capabilities.supports_image_generation);
    assert_eq!(capabilities.context_window, Some(262_144));

    let path_selection = qwen_entry.download_path_selection();
    assert!(path_selection.includes("config.json"));
    assert!(path_selection.includes("model.safetensors.index.json"));
    assert!(path_selection.includes("model-00001-of-00005.safetensors"));
    assert!(path_selection.includes("model-00005-of-00005.safetensors"));
    assert!(path_selection.includes("optiq/optiq_vision.safetensors"));
    assert!(path_selection.includes("optiq/mtp.safetensors"));
    assert!(path_selection.includes("tokenizer.json"));
}
