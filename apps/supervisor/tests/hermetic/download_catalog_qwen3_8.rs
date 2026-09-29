//! Bundled-catalog contract for the Qwen 3.8 35B-A3B Distill OptiQ 6-bit offer.

use astronomical_supervisor::{DownloadCatalog, DownloadCatalogFamily};

#[test]
fn should_offer_qwen_3_8_35b_a3b_distill_optiq_6bit_from_the_bundled_catalog() {
    let download_catalog = DownloadCatalog::load_bundled()
        .expect("the bundled production catalog should remain valid");
    let qwen_entry = download_catalog
        .entries()
        .iter()
        .find(|entry| entry.huggingface_id() == "funnygeeker/Qwen3.8-35B-A3B-Distill-oQ6e-mtp")
        .expect(
            "the release catalog should include the Qwen 3.8 35B-A3B Distill OptiQ 6-bit offer",
        );

    // The pinned revision is the contract: it must match the reviewed Hub tree,
    // not whatever the repository head happens to be.
    assert_eq!(
        qwen_entry.revision(),
        "0caaeb0e5482c37b5b1749d04b7dd1893239090d"
    );
    assert_eq!(qwen_entry.family(), DownloadCatalogFamily::Qwen3_5);
    assert_eq!(qwen_entry.approximate_size_bytes(), 30_106_500_432);
    assert_eq!(qwen_entry.upstream_license(), Some("Apache-2.0"));
    assert_eq!(
        qwen_entry.quantization_label(),
        Some("OptiQ mixed-precision 6/8-bit (affine, groups 64/128)")
    );

    let capabilities = qwen_entry.capabilities();
    assert!(capabilities.supports_reasoning);
    assert!(capabilities.supports_vision);
    assert!(capabilities.supports_tool_calls);
    assert!(!capabilities.supports_embeddings);
    assert!(!capabilities.supports_image_generation);
    assert_eq!(capabilities.context_window, Some(262_144));

    // This offer ships without an explicit path selection, so every Hub file
    // belongs to the executable package: the six language shards carry the
    // language model, the embedded vision tower, and the MTP head together.
    let path_selection = qwen_entry.download_path_selection();
    assert!(path_selection.includes("config.json"));
    assert!(path_selection.includes("model.safetensors.index.json"));
    assert!(path_selection.includes("model-00001-of-00006.safetensors"));
    assert!(path_selection.includes("model-00006-of-00006.safetensors"));
    assert!(path_selection.includes("tokenizer.json"));
    assert!(path_selection.includes("tokenizer_config.json"));
    assert!(path_selection.includes("generation_config.json"));
    assert!(path_selection.includes("chat_template.jinja"));
    assert!(path_selection.includes("preprocessor_config.json"));
}
