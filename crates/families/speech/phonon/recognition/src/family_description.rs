//! How discovery recognizes the Phonon speech family and reads one of its
//! folders, as one description the family list names (issue #2473, Step 5).

use std::path::Path;

use model_discovery::*;

use crate::*;

/// How discovery recognizes and reads a folder of this family.
pub const PHONON_FAMILY_DESCRIPTION: ModelFamilyDescription =
    ModelFamilyDescription::recognized_by_marker_file(
        PHONON_MODEL_FAMILY,
        ModelKind::Speech,
        phonon::classifies_directory,
        discover_phonon_model,
    );

/// Projects a Phonon folder into the neutral discovery contract.
pub fn discover_phonon_model(model_directory: &Path, model_id: &str) -> Option<DiscoveredModel> {
    discover_speech_model(
        model_directory,
        model_id,
        phonon_speech_capabilities(model_directory)?,
        PHONON_WEIGHTS_FILE_NAME,
    )
}
