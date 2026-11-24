//! How the worker builds any image family's unloaded engine (issue #2473,
//! Step 5): the family names its folder check, its provenance and its
//! engine builder, and the worker runs the same steps for every family, so
//! a new image family adds no worker code.
//!
//! The folder check runs on a blocking task because it reads the whole
//! folder; the engine it leads to is not `Send` and is built afterwards.

use std::path::{Path, PathBuf};

use model_discovery::ImageDirectoryEvidence;
use model_primitives::FamilyEngineSettings;

use crate::image_engine_shell::{ImageEngineShell, ImageFamily};

/// Why a selected image folder cannot be served. The worker words the reason
/// with the family's name and leaves out which detail of the folder changed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ImageArtifactRefusal {
    /// The folder no longer matches what discovery accepted.
    FailedVerification,
    /// The folder is sound, but it is not the model id or revision the
    /// daemon selected.
    UnsupportedIdentity,
}

/// The folder check discovery runs for a family.
pub type VerifyImageDirectory<Family> =
    fn(
        &Path,
    ) -> Result<ImageDirectoryEvidence, <Family as ImageFamilyConstruction>::DirectoryError>;

/// The family's provenance from a verified folder's evidence.
pub type ImageProvenanceFromEvidence<Family> =
    fn(ImageDirectoryEvidence) -> <Family as ImageFamilyConstruction>::Provenance;

/// The family's unloaded engine over a verified folder.
pub type BuildImageEngine<Family> = fn(
    PathBuf,
    <Family as ImageFamilyConstruction>::Provenance,
    &FamilyEngineSettings,
) -> ImageEngineShell<Family>;

/// What one image family names for the worker to build its engine.
pub trait ImageFamilyConstruction: ImageFamily + Sized + 'static {
    /// The family's record of which reviewed build a folder holds.
    type Provenance;
    /// Why the family's folder check refused a folder; the worker never
    /// shows it, so it cannot reveal which local detail changed.
    type DirectoryError;
    const VERIFY_DIRECTORY: VerifyImageDirectory<Self>;
    const PROVENANCE: ImageProvenanceFromEvidence<Self>;
    /// Nothing is read and no MLX work runs until `load`.
    const BUILD_ENGINE: BuildImageEngine<Self>;
}

/// Checks the folder and that it holds `model_id` at `artifact_revision`,
/// which every image family's folder check ends with.
///
/// # Errors
/// Returns the refusal the worker words for the client.
pub fn verified_image_artifact<Family: ImageFamilyConstruction>(
    model_directory_path: &Path,
    model_id: &str,
    artifact_revision: &str,
) -> Result<ImageDirectoryEvidence, ImageArtifactRefusal> {
    let verified_evidence = (Family::VERIFY_DIRECTORY)(model_directory_path)
        .map_err(|_| ImageArtifactRefusal::FailedVerification)?;
    if model_id == verified_evidence.canonical_model_id
        && artifact_revision == verified_evidence.revision
    {
        Ok(verified_evidence)
    } else {
        Err(ImageArtifactRefusal::UnsupportedIdentity)
    }
}

/// The family's unloaded engine over a verified folder.
pub fn image_engine_from_evidence<Family: ImageFamilyConstruction>(
    family_engine_settings: &FamilyEngineSettings,
    model_directory_path: PathBuf,
    verified_evidence: ImageDirectoryEvidence,
) -> ImageEngineShell<Family> {
    (Family::BUILD_ENGINE)(
        model_directory_path,
        (Family::PROVENANCE)(verified_evidence),
        family_engine_settings,
    )
    .with_floor_raise_limit_bytes(family_engine_settings.floor_raise_limit_bytes)
}
