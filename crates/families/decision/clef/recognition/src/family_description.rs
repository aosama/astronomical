//! How discovery recognizes the Clef decision family and reads one of its
//! folders, as one description the family list names (issue #2473, Step 5).

use std::path::Path;

use model_discovery::*;

use crate::*;

/// How discovery recognizes and reads a folder of this family.
pub const CLEF_FAMILY_DESCRIPTION: ModelFamilyDescription =
    ModelFamilyDescription::recognized_by_marker_file(
        CLEF_MODEL_FAMILY,
        ModelKind::Decision,
        clef::classifies_directory,
        discover_clef_model,
    );

/// Projects a Clef folder into the neutral discovery contract.
pub fn discover_clef_model(model_directory: &Path, model_id: &str) -> Option<DiscoveredModel> {
    discover_marked_decision_model(
        model_directory,
        model_id,
        clef::CLEF_MODEL_FAMILY,
        clef::discover_model_metadata(model_directory)?,
    )
}
