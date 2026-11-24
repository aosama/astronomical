//! How discovery recognizes the Kokoro speech family and reads one of its
//! folders, as one description the family list names (issue #2473, Step 5).

use std::path::Path;

use model_discovery::*;

use crate::*;

/// How discovery recognizes and reads a folder of this family.
pub const KOKORO_FAMILY_DESCRIPTION: ModelFamilyDescription =
    ModelFamilyDescription::recognized_by_marker_file(
        KOKORO_MODEL_FAMILY,
        ModelKind::Speech,
        kokoro::classifies_directory,
        discover_kokoro_model,
    );

/// Projects a Kokoro folder into the neutral discovery contract.
pub fn discover_kokoro_model(model_directory: &Path, model_id: &str) -> Option<DiscoveredModel> {
    discover_speech_model(
        model_directory,
        model_id,
        kokoro_speech_capabilities(model_directory)?,
        KOKORO_WEIGHTS_FILE_NAME,
    )
}
