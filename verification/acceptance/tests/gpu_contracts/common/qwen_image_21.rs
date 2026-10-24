//! Shared journey helpers for the Qwen-Image-2.1 acceptance tests.
//!
//! The MLX runtime is process-global: `MlxRuntime::initialize` accepts a second call only when
//! the memory limits are byte-identical, so every Qwen-Image-2.1 GPU journey in one test binary
//! must derive the SAME limit. This module resolves the artifact through the explicit test-run
//! model directory and computes the limit from every component weight file plus fixed activation
//! headroom — adapting to any artifact revision and any machine, never a developer path.

use std::path::PathBuf;

use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};

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
/// The selected directory contains component `model.safetensors` files and the
/// `processor/` tokenizer.
pub(crate) fn artifact_directory() -> PathBuf {
    astronomical_test_harness::test_run::TestRunContext::current().required_model_directory()
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
