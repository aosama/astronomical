//! The worker of Astronomical and the wiring between model families
//! (issue #2340): `engine_backed_worker` turns worker commands into engine
//! calls for every kind of model, and `model_family_runtime` chooses and
//! loads the family a model folder belongs to.
//!
//! The families and everything they share live in crates below this one:
//! the model primitives, artifact validation, the engine core, the Qwen 3.5
//! family, the decision and embedding models, image generation and speech.
//! This crate re-exports each of their public surfaces whole, so the worker
//! binaries and the acceptance tests reach every engine through one crate.

// Unsafe is denied workspace-wide, and this crate has none of its own. The one
// exception, the macOS accounting syscall behind performance attribution,
// lives in model-primitives.

// Artifact validation is shared by every model family and lives in its own
// crate (issue #2340); the alias keeps `crate::artifact_validation` paths.
pub(crate) use ::artifact_validation;

mod engine_backed_worker;

// Speech lives in its own crates (issue #2340, #2473); the alias keeps each
// `crate::speech_engine` path in this crate resolving.
pub(crate) use ::speech::speech_engine;

// The engine core (the inference contract and its errors, MLX memory, the
// astro_decoder_state_cache, the generation processor and structured
// generation) sits below the model families in its own crate (issue #2340).
// Its public surface, modules included, is re-exported whole, so every
// `crate::memory`, `crate::inference_engine` and root path here resolves.
pub use ::engine_core::*;

// Image generation lives in its own crates (issue #2340, #2473): the shell
// and one crate per family. Their public surfaces, and the family crates as
// `flux2_klein` and `qwen_image_21` modules, are re-exported so the worker,
// this crate's own paths and the acceptance tests keep resolving.
pub use ::flux2_klein;
pub use ::flux2_klein::*;
pub use ::image_generation::*;
pub use ::qwen_image_21;
pub use ::qwen_image_21::*;

// The model primitives every model family shares (issue #2340); their
// public surface, modules included, is re-exported whole.
pub use ::model_primitives::*;

// The Qwen 3.5 family (issue #2340): the core, the resident and streaming
// engines, Prism Hadamard, speculative decoding, expert paging, the kernel
// capability probes and the astro_ssd_prompt_cache. Its public surface,
// modules included, is re-exported whole.
pub use ::qwen35::*;

// The decision shell (issues #2340, #2473): the decision and embedding seams
// the worker drives and what the decision families share, re-exported whole.
// Each decision family is a crate of its own; the worker reaches its engine
// through the names below.
pub use ::decision_models::*;
#[cfg(feature = "direct-mlx")]
pub use clef::{ClefDecisionEngine, ClefDecisionFamily, clef_sequence_token_ids};
#[cfg(feature = "direct-mlx")]
pub use kev::{KevDecisionEngine, KevDecisionFamily, KevRowLayout, kev_question_row_layouts};
#[cfg(feature = "direct-mlx")]
pub use laya::{LayaDecisionEngine, LayaDecisionFamily, laya_question_sequences};
#[cfg(feature = "direct-mlx")]
pub use modernbert::{
    EncodedEmbeddingInput, ModernBertEmbeddingEngine, ModernBertEmbeddingFamily,
    encode_embedding_input,
};

#[doc(hidden)]
pub use ::artifact_validation::validate_required_file_for_tests;

pub use ::artifact_validation::{ArtifactValidationError, TensorDtype, TensorProfile};

#[doc(hidden)]
pub use ::artifact_validation::{
    RawSafetensorsInventoryForTests, RawSafetensorsTensorDescriptorForTests,
    validate_safetensors_required_profiles_for_tests,
};

pub use ipc_protocol::ExpertMemoryMode;

pub use engine_backed_worker::{
    DirectEngines, EngineBackedWorker, FactoryEngines, ModelFactory, ModelFactoryRuntime,
    WorkerEngines, WorkerRuntimeError,
};

#[cfg(feature = "direct-mlx")]
#[cfg(feature = "direct-mlx")]
pub use kokoro::{KokoroSpeechEngine, KokoroSpeechFamily, KokoroSpeechTrace, KokoroSpeechTracing};

#[cfg(feature = "direct-mlx")]
pub use phonon::{PhononSpeechEngine, PhononSpeechFamily};

// The speech engine shell and its family trait, which the worker names to
// build any speech family's engine by its family type.
#[cfg(feature = "direct-mlx")]
pub use ::speech::speech_models::speech_engine_shell::{SpeechEngineShell, SpeechFamily};

pub use speech_engine::{
    SpeechEngine, SpeechEngineLoadResult, SpeechEngineStep, SpeechUnavailableEngine, StartedSpeech,
};

// Restored public surface: these items are the API the crate's own integration

// tests consume (they verify the public seam); everything else that only the

// beside-source unit tests exercised stays crate-private.

pub use crate::artifact_validation::tensor_inventory::{
    TensorDeclarationOrigin, TensorInventory, TensorInventoryError, TensorLocation,
    TensorSemanticRole, TensorSourceId,
};

pub use crate::artifact_validation::types::RequiredFileProfile;

pub use crate::artifact_validation::validated_artifact::ValidatedWeightsFile;

// Qwen-Image-2.1: the family's pure contracts, artifact validation, and geometry constants. The

// MLX components (text encoder, transformer, VAE decoder, pipeline) follow below under `direct-mlx`.

/// Validates a safetensors shard where some tensors have strict dtype/shape profiles
/// and remaining tensors are accepted by name only.
///
/// Exposed for integration tests that build synthetic multi-tensor shard fixtures.
pub fn validate_bounded_safetensors_with_partial_profiles(
    weights_file: &std::fs::File,

    file_size_bytes: u64,

    weights_file_name: &str,

    profiled_tensor_profiles: &[TensorProfile],

    accepted_extra_tensor_names: &std::collections::HashSet<&str>,
) -> Result<artifact_validation::PartialProfileMetadata, ArtifactValidationError> {
    artifact_validation::validate_bounded_safetensors_with_partial_profiles(
        weights_file,
        file_size_bytes,
        weights_file_name,
        profiled_tensor_profiles,
        accepted_extra_tensor_names,
    )
}
