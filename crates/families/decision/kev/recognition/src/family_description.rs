//! How discovery recognizes the Kev decision family and reads one of its
//! folders, as one description the family list names (issue #2473, Step 5).

use std::path::Path;

use model_discovery::{
    DiscoveredModel, ModelFamilyDescription, ModelKind, discover_marked_decision_model,
};

use crate::{KEV_MODEL_FAMILY, kev};

/// How discovery recognizes and reads a folder of this family.
pub const KEV_FAMILY_DESCRIPTION: ModelFamilyDescription =
    ModelFamilyDescription::recognized_by_marker_file(
        KEV_MODEL_FAMILY,
        ModelKind::Decision,
        kev::classifies_directory,
        discover_kev_model,
    );

/// Projects a Kev folder into the neutral discovery contract.
pub fn discover_kev_model(model_directory: &Path, model_id: &str) -> Option<DiscoveredModel> {
    discover_marked_decision_model(
        model_directory,
        model_id,
        kev::KEV_MODEL_FAMILY,
        kev::discover_model_metadata(model_directory)?,
    )
}
