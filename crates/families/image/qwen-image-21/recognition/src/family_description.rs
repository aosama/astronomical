//! How discovery recognizes the Qwen-Image-2.1 image family and reads one of its
//! folders, as one description the family list names (issue #2473, Step 5).

use std::path::Path;

use model_discovery::*;

use crate::*;

/// How discovery recognizes and reads a folder of this family.
pub const QWEN_IMAGE_21_FAMILY_DESCRIPTION: ModelFamilyDescription =
    ModelFamilyDescription::recognized_by_pipeline_index(
        QWEN_IMAGE_21_MODEL_FAMILY,
        ModelKind::ImageGeneration,
        qwen_image_21::classifies_pipeline_index,
        discover_qwen_image_21_model,
    );

/// Projects a verified folder of the family into the neutral discovery contract.
pub fn discover_qwen_image_21_model(
    model_directory: &Path,
    _folder_model_id: &str,
) -> Option<DiscoveredModel> {
    let verified_evidence = qwen_image_21::verify_model_directory(model_directory).ok()?;
    Some(discovered_image_model(
        model_directory,
        QWEN_IMAGE_21_MODEL_FAMILY,
        verified_evidence,
    ))
}
