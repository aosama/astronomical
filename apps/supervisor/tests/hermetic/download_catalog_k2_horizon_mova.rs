//! Bundled-catalog contract for the K2 Horizon MoVA 36B A4B 4-bit offer.

use astronomical_supervisor::{DownloadCatalog, DownloadCatalogFamily};

#[test]
fn should_offer_the_stacked_affine_k2_horizon_mova_variant_from_the_bundled_catalog() {
    let download_catalog = DownloadCatalog::load_bundled()
        .expect("the bundled production catalog should remain valid");
    let k2_entries = download_catalog
        .entries()
        .iter()
        .filter(|entry| entry.family() == DownloadCatalogFamily::K2HorizonMoVA)
        .collect::<Vec<_>>();

    assert_eq!(
        k2_entries.len(),
        1,
        "the catalog offers one executable K2 Horizon MoVA conversion"
    );
    let k2_entry = k2_entries[0];
    assert_eq!(
        k2_entry.huggingface_id(),
        "abenzerps/K2-Horizon-MoVA-36B-A4B-MLX-4bit"
    );
    assert_eq!(
        k2_entry.revision(),
        "0c576733b69e8ca2d7a0292d0f01ee1d955bb9b5"
    );
    assert_eq!(k2_entry.approximate_size_bytes(), 21_102_192_601);
    assert_eq!(k2_entry.upstream_license(), Some("Apache-2.0"));
    assert_eq!(
        k2_entry.quantization_label(),
        Some("4-bit affine (group 64)")
    );

    let capabilities = k2_entry.capabilities();
    assert!(capabilities.supports_reasoning);
    assert!(capabilities.supports_tool_calls);
    assert!(!capabilities.supports_vision);
    assert!(!capabilities.supports_embeddings);
    assert!(!capabilities.supports_image_generation);
    assert_eq!(capabilities.context_window, Some(524_288));

    let path_selection = k2_entry.download_path_selection();
    assert!(path_selection.includes("config.json"));
    assert!(path_selection.includes("model.safetensors.index.json"));
    assert!(path_selection.includes("tokenizer.json"));
    assert!(path_selection.includes("chat_template.jinja"));
    assert!(path_selection.includes("model-00001-of-00048.safetensors"));
    assert!(path_selection.includes("model-00048-of-00048.safetensors"));
    assert!(
        !path_selection.includes("README.md"),
        "the executable payload must not ship repository extras"
    );
    assert!(
        !path_selection.includes("k2_horizon_mova_mlx.py"),
        "the executable payload must not ship checkpoint Python"
    );
    assert!(
        !path_selection.includes("assets/k2-horizon-mova-36b-a4b-benchmarks.png"),
        "the executable payload must not ship marketing assets"
    );
}
