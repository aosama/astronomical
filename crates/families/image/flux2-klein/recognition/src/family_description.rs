//! How discovery recognizes the FLUX.2 Klein image family and reads one of its
//! folders, as one description the family list names (issue #2473, Step 5).

use std::path::Path;

use model_discovery::{DiscoveredModel, ModelFamilyDescription, ModelKind, discovered_image_model};

use crate::{FLUX2_KLEIN_MODEL_FAMILY, flux2_klein, flux2_klein_mlx_4bit};

/// How discovery recognizes and reads a folder of this family.
pub const FLUX2_KLEIN_FAMILY_DESCRIPTION: ModelFamilyDescription =
    ModelFamilyDescription::recognized_by_pipeline_index(
        FLUX2_KLEIN_MODEL_FAMILY,
        ModelKind::ImageGeneration,
        flux2_klein::classifies_pipeline_index,
        discover_flux2_klein_model,
    )
    .also_recognized_by_marker_file(
        flux2_klein_mlx_4bit::classifies_directory,
        flux2_klein_mlx_4bit::FLUX2_KLEIN_MLX_4BIT_TRANSFORMER_INDEX_FILE_NAME,
    )
    .also_marks_pipeline_root(flux2_klein_mlx_4bit::has_build_marker);

/// Projects a verified folder of the family into the neutral discovery contract.
pub fn discover_flux2_klein_model(
    model_directory: &Path,
    _folder_model_id: &str,
) -> Option<DiscoveredModel> {
    let verified_evidence = flux2_klein::verify_model_directory(model_directory).ok()?;
    Some(discovered_image_model(
        model_directory,
        FLUX2_KLEIN_MODEL_FAMILY,
        verified_evidence,
    ))
}
