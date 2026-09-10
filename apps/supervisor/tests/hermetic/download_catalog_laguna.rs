//! Bundled-catalog contract for the Laguna XS 2.1 family offers.

use astronomical_supervisor::{DownloadCatalog, DownloadCatalogFamily};

#[test]
fn should_offer_the_validated_laguna_variants_from_the_bundled_catalog() {
    let download_catalog = DownloadCatalog::load_bundled()
        .expect("the bundled production catalog should remain valid");
    let laguna_entries = download_catalog
        .entries()
        .iter()
        .filter(|entry| entry.family() == DownloadCatalogFamily::Laguna)
        .collect::<Vec<_>>();

    let expected_laguna_entries = [
        (
            "mlx-community/Laguna-XS-2.1-6bit",
            "5f60b7511ea0be481a68be1500baf15b0bcccba6",
            27_185_160_853,
        ),
        (
            "mlx-community/Laguna-XS-2.1-5bit",
            "f9990bed9f13df5089804c44a6ef9ea2589fe5df",
            23_007_413_034,
        ),
        (
            "mlx-community/Laguna-XS-2.1-4bit",
            "c42e0a8f8d504ceacde015a535dcb286d65c8799",
            18_829_665_100,
        ),
    ];
    assert_eq!(
        laguna_entries
            .iter()
            .map(|entry| (
                entry.huggingface_id(),
                entry.revision(),
                entry.approximate_size_bytes()
            ))
            .collect::<Vec<_>>(),
        expected_laguna_entries
            .iter()
            .map(|(huggingface_id, revision, approximate_size_bytes)| (
                *huggingface_id,
                *revision,
                *approximate_size_bytes
            ))
            .collect::<Vec<_>>(),
        "the bundled catalog must offer the validated Laguna variants pinned to immutable revisions"
    );

    for laguna_entry in &laguna_entries {
        let capabilities = laguna_entry.capabilities();
        assert!(
            capabilities.supports_reasoning,
            "{}",
            laguna_entry.huggingface_id()
        );
        assert!(
            capabilities.supports_tool_calls,
            "{}",
            laguna_entry.huggingface_id()
        );
        assert!(
            !capabilities.supports_vision,
            "Laguna XS 2.1 is text-only: {}",
            laguna_entry.huggingface_id()
        );
        assert_eq!(
            capabilities.context_window,
            Some(262_144),
            "{}",
            laguna_entry.huggingface_id()
        );
        assert_eq!(
            laguna_entry.upstream_license(),
            Some("OpenMDW-1.1"),
            "{}",
            laguna_entry.huggingface_id()
        );

        let path_selection = laguna_entry.download_path_selection();
        assert!(path_selection.includes("config.json"));
        assert!(path_selection.includes("model.safetensors.index.json"));
        assert!(path_selection.includes("tokenizer.json"));
        assert!(path_selection.includes("chat_template.jinja"));
        assert!(
            !path_selection.includes("README.md"),
            "the executable payload must not ship repository extras: {}",
            laguna_entry.huggingface_id()
        );
        assert!(
            !path_selection.includes(".eval_results/swe-bench_verified.yaml"),
            "the executable payload must not ship evaluation metadata: {}",
            laguna_entry.huggingface_id()
        );
    }
}
