//! Bundled-catalog contract for the text-only Ornith 1.5 35B-A3B MLX offer.

use astronomical_supervisor::{DownloadCatalog, DownloadCatalogFamily};

#[test]
fn should_offer_ornith_1_5_35b_a3b_mlx_6bit_from_the_bundled_catalog() {
    let download_catalog = DownloadCatalog::load_bundled()
        .expect("the bundled production catalog should remain valid");
    let ornith_entry = download_catalog
        .entries()
        .iter()
        .find(|entry| entry.huggingface_id() == "ornith-ai/Ornith-1.5-35B-A3B-MLX-6bit")
        .expect("the release catalog should include the Ornith MLX 6-bit offer");

    assert_eq!(
        ornith_entry.revision(),
        "585b7867b0517980293ece857b26d64e84491352"
    );
    assert_eq!(ornith_entry.family(), DownloadCatalogFamily::Qwen3_5);
    assert_eq!(ornith_entry.upstream_license(), Some("MIT"));
    assert_eq!(
        ornith_entry.quantization_label(),
        Some("6-bit affine (group 64)")
    );
    assert!(
        ornith_entry
            .architecture_summary()
            .is_some_and(|summary| summary.contains("256 experts"))
    );
    assert!(
        ornith_entry
            .architecture_summary()
            .is_some_and(|summary| summary.contains("8 routed per token"))
    );

    let capabilities = ornith_entry.capabilities();
    assert!(capabilities.supports_reasoning);
    assert!(capabilities.supports_tool_calls);
    assert!(!capabilities.supports_vision);
    assert!(!capabilities.supports_embeddings);
    assert!(!capabilities.supports_image_generation);
    assert_eq!(capabilities.context_window, Some(262_144));

    let path_selection = ornith_entry.download_path_selection();
    for shard_index in 1..=6 {
        assert!(path_selection.includes(&format!("model-{shard_index:05}-of-00006.safetensors")));
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
