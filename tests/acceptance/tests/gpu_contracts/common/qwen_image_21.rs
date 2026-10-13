//! Shared journey helpers for the Qwen-Image-2.1 acceptance tests.
//!
//! The MLX runtime is process-global: `MlxRuntime::initialize` accepts a second call only when
//! the memory limits are byte-identical, so every Qwen-Image-2.1 GPU journey in one test binary
//! must derive the SAME limit. This module resolves the artifact through the shared environment
//! variable and computes the limit from every component weight file plus fixed activation
//! headroom — adapting to any artifact revision and any machine, never a developer path.

use std::path::PathBuf;

use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};

/// Environment variable pointing at the installed artifact root (the
/// `models--.../snapshots/<sha>` directory containing `transformer/` and `vae/`).
pub(crate) const QWEN_IMAGE_21_ARTIFACT_ENV: &str = "ASTRONOMICAL_QWEN_IMAGE_21_ARTIFACT_DIRECTORY";

/// Activation headroom above the component weights. A render stages the components — the text
/// encoder and transformer release before the VAE decodes — but a journey that loads all three
/// before releasing still has to fit, and the 1024×1024 decode's staged f32 activations are the
/// peak below that. Two gigabytes covers both on any artifact revision.
const ACTIVATION_HEADROOM_BYTES: usize = 2 * 1024 * 1024 * 1024;
/// The MLX allocator cache stays small: journeys run once and release, so there is no reuse
/// worth banking on, and the memory ceiling is the real limit.
const ALLOCATOR_CACHE_LIMIT_BYTES: usize = 8 * 1024 * 1024;

/// The effective MLX ceiling every Qwen-Image-2.1 journey shares.
///
/// The serving engine constructs its own process-global runtime from this number, so exposing
/// it — rather than only the initialized runtime — keeps the engine journey byte-identical with
/// the pipeline journeys that share this process.
pub(crate) fn journey_effective_ceiling_bytes() -> usize {
    ["transformer", "vae", "text_encoder"]
        .iter()
        .map(|component| {
            usize::try_from(
                std::fs::metadata(component_weights_path(component))
                    .unwrap_or_else(|error| {
                        panic!("the {component} weights should be readable: {error}")
                    })
                    .len(),
            )
            .expect("component weight size should fit usize")
        })
        .sum::<usize>()
        + ACTIVATION_HEADROOM_BYTES
}

/// The allocator cache limit every Qwen-Image-2.1 journey shares.
pub(crate) const JOURNEY_ALLOCATOR_CACHE_LIMIT_BYTES: usize = ALLOCATOR_CACHE_LIMIT_BYTES;

/// Resolves the artifact root every Qwen-Image-2.1 journey shares.
///
/// The panic spells out what a valid root contains because journeys read different parts of it:
/// component `model.safetensors` files, the `processor/` tokenizer, or both.
pub(crate) fn artifact_directory() -> PathBuf {
    match std::env::var_os(QWEN_IMAGE_21_ARTIFACT_ENV) {
        Some(value) if !value.is_empty() => PathBuf::from(value),
        _ => panic!(
            "set {QWEN_IMAGE_21_ARTIFACT_ENV} to the installed Qwen-Image-2.1 artifact root \
             (the models--mlx-community--Qwen-Image-2.1-MLX-4bit snapshots directory containing \
             transformer/, vae/, text_encoder/, and processor/), not a hardcoded developer \
             path"
        ),
    }
}

/// The `model.safetensors` path of one artifact component (`"transformer"` or `"vae"`).
pub(crate) fn component_weights_path(component: &str) -> PathBuf {
    let weights_path = artifact_directory()
        .join(component)
        .join("model.safetensors");
    assert!(
        weights_path.exists(),
        "expected the Qwen-Image-2.1 {component} weights at {}",
        weights_path.display()
    );
    weights_path
}

/// The process-wide journey runtime, identical across every Qwen-Image-2.1 GPU journey.
///
/// A render can have all three components resident at once (text encoder, transformer, VAE
/// decoder), so the cap covers their combined weight bytes plus activation headroom; the
/// per-component journeys only ever allocate what they use against that cap.
pub(crate) fn shared_journey_runtime() -> MlxRuntime {
    MlxRuntime::initialize(
        MlxMemoryLimits::new(
            journey_effective_ceiling_bytes(),
            JOURNEY_ALLOCATOR_CACHE_LIMIT_BYTES,
        )
        .expect("the Qwen-Image-2.1 journey memory limits should be valid"),
    )
    .expect("the pinned MLX runtime should initialize")
}
