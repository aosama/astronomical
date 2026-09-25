//! Bundled-catalog contract for the Qwen-Image-2.1 family offer.

use astronomical_supervisor::{DownloadCatalog, DownloadCatalogFamily};

#[test]
fn should_package_qwen_image_2_1_with_only_its_executable_pipeline_graph() {
    let download_catalog = DownloadCatalog::load_bundled()
        .expect("the bundled production catalog should remain valid");
    let qwen_image_entry = download_catalog
        .entries()
        .iter()
        .find(|entry| entry.family() == DownloadCatalogFamily::QwenImage21)
        .expect("the release catalog should include the executable Qwen-Image-2.1 profile");

    // The pinned revision is the contract: it must match the reviewed 4-bit MLX conversion
    // that the acceptance journeys render with, not whatever the repository head happens to be.
    assert_eq!(
        qwen_image_entry.huggingface_id(),
        "mlx-community/Qwen-Image-2.1-MLX-4bit"
    );
    assert_eq!(
        qwen_image_entry.revision(),
        "4db4e8c0c0e7a1debf0320415bec8388e888494c"
    );
    assert!(qwen_image_entry.capabilities().supports_image_generation);
    assert!(!qwen_image_entry.capabilities().supports_reasoning);
    assert!(!qwen_image_entry.capabilities().supports_embeddings);
    // Keep the authored size within a decimal band so re-quantizations do not silently
    // invalidate the preflight disk estimate.
    assert!(qwen_image_entry.approximate_size_bytes() > 9_000_000_000);
    assert!(qwen_image_entry.approximate_size_bytes() < 12_000_000_000);
    for required_path in [
        "model_index.json",
        "processor/tokenizer.json",
        "processor/preprocessor_config.json",
        "scheduler/scheduler_config.json",
        "text_encoder/config.json",
        "text_encoder/model.safetensors",
        "transformer/config.json",
        "transformer/model.safetensors",
        "vae/config.json",
        "vae/model.safetensors",
    ] {
        assert!(
            qwen_image_entry
                .download_path_selection()
                .includes(required_path),
            "the Qwen-Image-2.1 offer must download {required_path}"
        );
    }
    assert!(
        !qwen_image_entry
            .download_path_selection()
            .includes("qwen-image-2.1.safetensors")
    );
    assert!(
        !qwen_image_entry
            .download_path_selection()
            .includes("editing.jpg")
    );
    assert!(
        qwen_image_entry
            .download_path_selection()
            .includes("README.md")
    );
}
