//! Builds the speech engine of a folder discovery classified as Kokoro or
//! Phonon. The unloaded engine holds only its settings; the MLX runtime is
//! created inside `load()` on the worker runtime thread, because MLX handles
//! are not `Send` across task boundaries.

use std::path::PathBuf;

use astronomical_config::ModelFamily;
use astronomical_ipc_protocol::WorkerModelConfiguration;
use astronomical_model_serving::{KokoroSpeechEngine, PhononSpeechEngine, SpeechEngine};

/// The memory a speech engine is granted, as the factory holds it.
pub(super) struct SpeechMemoryGrant {
    pub(super) effective_mlx_memory_ceiling_bytes: usize,
    pub(super) allocator_cache_memory_limit_bytes: usize,
}

/// The unloaded speech engine for `model_family`, or the mismatch reason the
/// other forward-pass families give when a family and a configuration
/// disagree.
pub(super) fn unloaded_speech_engine(
    model_family: Option<ModelFamily>,
    model_directory_path: PathBuf,
    model_configuration: WorkerModelConfiguration,
    memory_grant: &SpeechMemoryGrant,
) -> Result<Box<dyn SpeechEngine>, String> {
    let WorkerModelConfiguration::Speech(speech_configuration) = model_configuration else {
        return Err(MISMATCHED_FAMILY_REASON.to_owned());
    };
    let model_id = speech_configuration.model_id;
    match model_family {
        Some(ModelFamily::Kokoro) => Ok(Box::new(KokoroSpeechEngine::new(
            model_id,
            model_directory_path,
            memory_grant.effective_mlx_memory_ceiling_bytes,
            memory_grant.allocator_cache_memory_limit_bytes,
        ))),
        Some(ModelFamily::Phonon) => Ok(Box::new(PhononSpeechEngine::new(
            model_id,
            model_directory_path,
            memory_grant.effective_mlx_memory_ceiling_bytes,
            memory_grant.allocator_cache_memory_limit_bytes,
        ))),
        _ => Err(MISMATCHED_FAMILY_REASON.to_owned()),
    }
}

const MISMATCHED_FAMILY_REASON: &str =
    "selected model configuration does not match its classified model family";

#[cfg(test)]
#[path = "speech_engine_creation_tests.rs"]
mod tests;
