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

use registered_model_families::REGISTERED_MODEL_FAMILIES;

/// Every registered family, in classification order.
pub static MODEL_FAMILY_REGISTRY: ModelFamilyRegistry =
    ModelFamilyRegistry::new(&REGISTERED_MODEL_FAMILIES);

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
