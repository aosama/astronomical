//! Bundled-catalog contract for the text-only Qwen 3.8 35B-A3B Distill 4-bit offer.

use astronomical_supervisor::{DownloadCatalog, DownloadCatalogFamily};

#[test]
fn should_offer_qwen_3_8_35b_a3b_distill_4bit_from_the_bundled_catalog() {
    let download_catalog = DownloadCatalog::load_bundled()
        .expect("the bundled production catalog should remain valid");
    let qwen_entry = download_catalog
        .entries()
        .iter()
        .find(|entry| entry.huggingface_id() == "nvythong/Qwen3.8-35B-A3B-Distill-mlx-4Bit")
        .expect("the release catalog should include the Qwen 3.8 35B-A3B Distill 4-bit offer");

    assert_eq!(
        qwen_entry.revision(),
        "220dfec3e716bbae93bd66b643e36b2ffe6853ff"
    );
    assert_eq!(qwen_entry.family(), DownloadCatalogFamily::Qwen3_5);
    // The size feeds the disk preflight, so it must track the pinned Hub tree.
    assert_eq!(qwen_entry.approximate_size_bytes(), 19_529_235_716);
    assert_eq!(qwen_entry.upstream_license(), Some("Apache-2.0"));
    assert_eq!(
        qwen_entry.quantization_label(),
        Some("4-bit affine (group 64), 8-bit router and shared-expert gates")
    );
    assert!(
        qwen_entry
            .architecture_summary()
            .is_some_and(|summary| summary.contains("256 experts"))
    );
    assert!(
        qwen_entry
            .architecture_summary()
            .is_some_and(|summary| summary.contains("text-only"))
    );

    let capabilities = qwen_entry.capabilities();
    assert!(capabilities.supports_reasoning);
    assert!(capabilities.supports_tool_calls);
    assert!(!capabilities.supports_vision);
    assert!(!capabilities.supports_embeddings);
    assert!(!capabilities.supports_image_generation);
    assert_eq!(capabilities.context_window, Some(262_144));

    // This offer ships without an explicit path selection, so every Hub file
    // belongs to the executable package: the four language shards carry the
    // text-only language model together with tokenizer and config metadata.
    let path_selection = qwen_entry.download_path_selection();
    for shard_index in 1..=4 {
        assert!(path_selection.includes(&format!("model-{shard_index:05}-of-00004.safetensors")));
    }
    for required_path in [
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
        "chat_template.jinja",
    ] {
        assert!(path_selection.includes(required_path), "{required_path}");
    }
}
