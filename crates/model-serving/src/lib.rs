// Unsafe is denied workspace-wide, and this crate has none of its own. The one
// exception, the macOS accounting syscall behind performance attribution, moved
// with that module into astronomical-gpu-building-blocks (issue #2340).

// Artifact validation is shared by every model family and lives in its own
// crate (issue #2340); the alias keeps `crate::artifact_validation` paths.
pub(crate) use astronomical_artifact_validation as artifact_validation;

mod embedding_engine;

mod engine_backed_worker;

mod model_family_runtime;

pub(crate) mod decision_engine;

// Speech lives in its own crate (issue #2340); the aliases keep each
// `crate::speech_*` path in this crate resolving.
pub(crate) use astronomical_speech::speech_engine;

#[cfg(feature = "direct-mlx")]
pub(crate) mod decision_models;

#[cfg(feature = "direct-mlx")]
pub(crate) mod modernbert;

#[cfg(feature = "direct-mlx")]
pub(crate) use astronomical_speech::speech_models;

// The engine core (the inference contract and its errors, MLX memory, the
// astro_decoder_state_cache, the generation processor and structured
// generation) sits below the model families in its own crate (issue #2340).
// Its public surface, modules included, is re-exported whole, so every
// `crate::memory`, `crate::inference_engine` and root path here resolves.
pub use astronomical_engine_core::*;

// Image generation lives in its own crate (issue #2340). Its public surface,
// the `flux2_klein`, `qwen_image_21` and `image_generation_engine` modules
// included, is re-exported whole, so the worker, this crate's own
// `crate::flux2_klein` paths and the acceptance tests keep resolving.
pub use astronomical_image_generation::*;

// The GPU building blocks every model family shares (issue #2340); their
// public surface, modules included, is re-exported whole.
pub use astronomical_gpu_building_blocks::*;

// The Qwen 3.5 family (issue #2340): the core, the resident and streaming
// engines, Prism Hadamard, speculative decoding, expert paging, the kernel
// capability probes and the astro_ssd_prompt_cache. Its public surface,
// modules included, is re-exported whole.
pub use astronomical_qwen35::*;

#[doc(hidden)]
pub use artifact_validation::validate_required_file_for_tests;

pub use artifact_validation::{ArtifactValidationError, TensorDtype, TensorProfile};

#[doc(hidden)]
pub use artifact_validation::{
    RawSafetensorsInventoryForTests, RawSafetensorsTensorDescriptorForTests,
    validate_safetensors_required_profiles_for_tests,
};

pub use astronomical_ipc_protocol::ExpertMemoryMode;

pub use decision_engine::{
    DecisionEngine, DecisionEngineLoadResult, DecisionEngineOutput, DecisionTrainingStep,
    DecisionUnavailableEngine,
};

#[cfg(feature = "direct-mlx")]
pub use decision_models::clef::ClefDecisionEngine;
#[cfg(feature = "direct-mlx")]
pub use decision_models::kev::{KevDecisionEngine, KevRowLayout};
#[cfg(feature = "direct-mlx")]
pub use decision_models::laya::LayaDecisionEngine;
#[cfg(feature = "direct-mlx")]
pub use decision_models::recorded_training::RecordedTrainingEngine;

pub use embedding_engine::{
    EmbeddingEngine, EmbeddingEngineLoadResult, EmbeddingEngineOutput, EmbeddingUnavailableEngine,
};

pub use engine_backed_worker::{
    EngineBackedWorker, ModelFactory, ModelFactoryRuntime, WorkerRuntimeError,
};

#[cfg(feature = "direct-mlx")]
pub use model_family_runtime::ModelFamilyInferenceEngine;

pub use model_family_runtime::{
    ModelFamilyGenerationProcessor, ModelFamilyImageEngine, ModelFamilyInferenceRequest,
    ModelFamilyRequestOutput,
};

#[cfg(feature = "direct-mlx")]
pub use modernbert::{EncodedEmbeddingInput, ModernBertEmbeddingEngine, encode_embedding_input};

#[cfg(feature = "direct-mlx")]
pub use speech_models::kokoro::{KokoroSpeechEngine, KokoroSpeechTrace};

#[cfg(feature = "direct-mlx")]
pub use speech_models::phonon::PhononSpeechEngine;

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
