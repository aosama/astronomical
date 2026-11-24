//! The list of model families Astronomical serves (issue #2473, Step 5), and
//! discovery with that list already passed in.
//!
//! `model-discovery` holds the rules every family shares and names no
//! family; this crate names them all, once, in classification order. The
//! functions below keep the names discovery always had, so the daemon, the
//! command line and `config` call them as before. Like discovery, this crate
//! never links MLX: the daemon recognizes folders through it.

use std::path::{Path, PathBuf};

use model_discovery::{
    ClassifiedModelArtifact, DiscoveredModelError, EffectiveModelDiscovery,
    EffectiveModelDiscoveryError, ModelDiscoveryDirectoryScan, ModelDiscoveryReport, ModelFamily,
    ModelFamilyClassificationError, ModelFamilyRegistry, ModelKind,
};

mod registered_model_families;

// Each family's own names (its constant, file names, folder checks),
// re-exported so `config` and the daemon keep one path to them.
pub use clef_recognition::{CLEF_FAMILY_DESCRIPTION, CLEF_MODEL_FAMILY, discover_clef_model};
pub use flux2_klein_recognition::{
    FLUX2_KLEIN_FAMILY_DESCRIPTION, FLUX2_KLEIN_MLX_4BIT_TRANSFORMER_INDEX_FILE_NAME,
    FLUX2_KLEIN_MODEL_FAMILY, Flux2KleinDirectoryVerificationError,
    classify_mflux_transformer_index_bytes, discover_flux2_klein_model,
    verify_flux2_klein_model_directory,
};
pub use kev_recognition::{KEV_FAMILY_DESCRIPTION, KEV_MODEL_FAMILY, discover_kev_model};
pub use kokoro_recognition::{
    KOKORO_CONFIGURATION_FILE_NAME, KOKORO_FAMILY_DESCRIPTION, KOKORO_MODEL_FAMILY,
    KOKORO_VOICES_DIRECTORY_NAME, KOKORO_WEIGHTS_FILE_NAME, discover_kokoro_model,
    is_kokoro_configuration, kokoro_speech_capabilities,
};
pub use laya_recognition::{
    LAYA_FAMILY_DESCRIPTION, LAYA_MODEL_FAMILY, discover_laya_bundled_checkpoints,
    discover_laya_model,
};
pub use modernbert_recognition::{
    MODERNBERT_FAMILY_DESCRIPTION, MODERNBERT_MODEL_FAMILY, discover_modernbert_model,
};
pub use phonon_recognition::{
    PHONON_CONFIGURATION_FILE_NAME, PHONON_DEFAULT_VOICE_FILE_NAME, PHONON_DEFAULT_VOICE_NAME,
    PHONON_FAMILY_DESCRIPTION, PHONON_MODEL_FAMILY, PHONON_TOKENIZER_FILE_NAME,
    PHONON_VOICES_DIRECTORY_NAME, PHONON_WEIGHTS_FILE_NAME, discover_phonon_model,
    is_phonon_configuration, phonon_speech_capabilities,
};
pub use qwen_image_21_recognition::{
    QWEN_IMAGE_21_FAMILY_DESCRIPTION, QWEN_IMAGE_21_MODEL_FAMILY,
    QwenImage21DirectoryVerificationError, discover_qwen_image_21_model,
    verify_qwen_image_21_model_directory,
};
pub use qwen35_recognition::{
    MINIMUM_SERVABLE_CONTEXT_WINDOW_TOKENS, PRISM_HADAMARD_QWEN35_FAMILY_DESCRIPTION,
    PRISM_HADAMARD_QWEN35_MODEL_FAMILY, QWEN3_5_FAMILY_DESCRIPTION, QWEN3_5_MODEL_FAMILY,
    context_window_tokens, discover_prism_hadamard_qwen35_model, discover_qwen3_5_model,
    required_shard_file_names,
};

use registered_model_families::REGISTERED_MODEL_FAMILIES;

/// Every registered family, in classification order.
pub static MODEL_FAMILY_REGISTRY: ModelFamilyRegistry =
    ModelFamilyRegistry::new(REGISTERED_MODEL_FAMILIES);

/// The registered family with this identifier.
#[must_use]
pub fn model_family_named(identifier: &str) -> Option<ModelFamily> {
    MODEL_FAMILY_REGISTRY.family_named(identifier)
}

/// The kind of model a family is, or `None` when nothing registers it.
#[must_use]
pub fn model_kind_of(model_family: ModelFamily) -> Option<ModelKind> {
    MODEL_FAMILY_REGISTRY.kind_of(model_family)
}

/// The first family that claims this `config.json` `model_type`, without
/// claiming the folder holds a servable model.
#[must_use]
pub fn model_family_of_model_type(model_type: Option<&str>) -> Option<ModelFamily> {
    MODEL_FAMILY_REGISTRY.family_of_model_type(model_type)
}

/// Classifies a selected artifact directory from the one file its family is
/// known by.
///
/// # Errors
/// Returns the reason the identifying file could not be read or parsed.
pub fn classify_model_directory(
    model_directory: &Path,
) -> Result<Option<ModelFamily>, ModelFamilyClassificationError> {
    model_discovery::classify_model_directory(&MODEL_FAMILY_REGISTRY, model_directory)
}

/// Classifies a Diffusers pipeline index document without a file-system read.
///
/// # Errors
/// Returns the pipeline document parse failure verbatim.
pub fn classify_pipeline_index_bytes(
    pipeline_index_bytes: &[u8],
) -> Result<Option<ModelFamily>, serde_json::Error> {
    model_discovery::classify_pipeline_index_bytes(&MODEL_FAMILY_REGISTRY, pipeline_index_bytes)
}

/// Discovers executable models under each directory.
///
/// # Errors
/// Returns a scan failure that makes the result unusable.
pub fn discover_models(
    model_directories: &[PathBuf],
) -> Result<Vec<ModelDiscoveryDirectoryScan>, DiscoveredModelError> {
    model_discovery::discover_models(&MODEL_FAMILY_REGISTRY, model_directories)
}

/// Discovers executable models, leaving out identities found under more
/// than one configured directory.
///
/// # Errors
/// Returns a scan failure that makes the result unusable.
pub fn discover_models_excluding_ambiguous_identities(
    model_directories: &[PathBuf],
) -> Result<ModelDiscoveryReport, DiscoveredModelError> {
    model_discovery::discover_models_excluding_ambiguous_identities(
        &MODEL_FAMILY_REGISTRY,
        model_directories,
    )
}

/// Finds classified model-family artifacts without advertising them as
/// executable models.
///
/// # Errors
/// Returns a scan failure of a configured directory.
pub fn discover_classified_model_artifacts(
    model_directories: &[PathBuf],
) -> Result<Vec<ClassifiedModelArtifact>, DiscoveredModelError> {
    model_discovery::discover_classified_model_artifacts(&MODEL_FAMILY_REGISTRY, model_directories)
}

/// Discovers executable models across the automatic Library root and the
/// authored roots, in the precedence the daemon serves.
///
/// # Errors
/// Returns a scan failure that makes the result unusable.
pub fn discover_effective_models(
    automatic_models_directory: &Path,
    configured_model_directories: &[PathBuf],
) -> Result<EffectiveModelDiscovery, EffectiveModelDiscoveryError> {
    model_discovery::discover_effective_models(
        &MODEL_FAMILY_REGISTRY,
        automatic_models_directory,
        configured_model_directories,
    )
}

/// Whether discovery, scanning `model_root`, would find a model written to
/// `model_folder`.
#[must_use]
pub fn is_discoverable_model_folder(model_root: &Path, model_folder: &Path) -> bool {
    model_discovery::is_discoverable_model_folder(&MODEL_FAMILY_REGISTRY, model_root, model_folder)
}
