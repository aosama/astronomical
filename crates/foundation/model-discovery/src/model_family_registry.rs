//! Every model family's description, in the order a folder is classified
//! (issue #2473, Step 5).
//!
//! One explicit list, rather than registration at link time, so a reader
//! sees every family in one place and a family left out fails the registry
//! tests instead of vanishing at run time. Within each way of recognizing a
//! folder (pipeline index, marker file, `model_type`) the first family that
//! claims it wins, so the order below is the classification order. Each
//! family declares its constant and its recognizers in its own module.

use super::artifact_discovery::{
    discover_clef_model, discover_flux2_klein_model, discover_kev_model, discover_kokoro_model,
    discover_laya_bundled_checkpoints, discover_laya_model, discover_modernbert_model,
    discover_phonon_model, discover_prism_hadamard_qwen35_model, discover_qwen_image_21_model,
    discover_qwen3_5_model,
};
use super::model_family_description::{ModelFamilyDescription, ModelKind};
use super::{
    CLEF_MODEL_FAMILY, FLUX2_KLEIN_MODEL_FAMILY, KEV_MODEL_FAMILY, KOKORO_MODEL_FAMILY,
    LAYA_MODEL_FAMILY, MODERNBERT_MODEL_FAMILY, PHONON_MODEL_FAMILY,
    PRISM_HADAMARD_QWEN35_MODEL_FAMILY, QWEN_IMAGE_21_MODEL_FAMILY, QWEN3_5_MODEL_FAMILY, clef,
    flux2_klein, flux2_klein_mlx_4bit, kev, kokoro, laya, modernbert, phonon,
    prism_hadamard_qwen35, qwen_image_21, qwen3_5,
};

pub(super) static REGISTERED_MODEL_FAMILIES: [ModelFamilyDescription; 10] = [
    ModelFamilyDescription::recognized_by_pipeline_index(
        FLUX2_KLEIN_MODEL_FAMILY,
        ModelKind::ImageGeneration,
        flux2_klein::classifies_pipeline_index,
        discover_flux2_klein_model,
    )
    .also_recognized_by_marker_file(
        flux2_klein_mlx_4bit::classifies_directory,
        flux2_klein_mlx_4bit::FLUX2_KLEIN_MLX_4BIT_TRANSFORMER_INDEX_FILE_NAME,
    ),
    ModelFamilyDescription::recognized_by_pipeline_index(
        QWEN_IMAGE_21_MODEL_FAMILY,
        ModelKind::ImageGeneration,
        qwen_image_21::classifies_pipeline_index,
        discover_qwen_image_21_model,
    ),
    ModelFamilyDescription::recognized_by_marker_file(
        KEV_MODEL_FAMILY,
        ModelKind::Decision,
        kev::classifies_directory,
        discover_kev_model,
    ),
    ModelFamilyDescription::recognized_by_marker_file(
        CLEF_MODEL_FAMILY,
        ModelKind::Decision,
        clef::classifies_directory,
        discover_clef_model,
    ),
    ModelFamilyDescription::recognized_by_marker_file(
        KOKORO_MODEL_FAMILY,
        ModelKind::Speech,
        kokoro::classifies_directory,
        discover_kokoro_model,
    ),
    ModelFamilyDescription::recognized_by_marker_file(
        PHONON_MODEL_FAMILY,
        ModelKind::Speech,
        phonon::classifies_directory,
        discover_phonon_model,
    ),
    ModelFamilyDescription::recognized_by_model_type(
        QWEN3_5_MODEL_FAMILY,
        ModelKind::Chat,
        qwen3_5::recognizes_model_type,
        discover_qwen3_5_model,
    ),
    ModelFamilyDescription::recognized_by_model_type(
        PRISM_HADAMARD_QWEN35_MODEL_FAMILY,
        ModelKind::Chat,
        prism_hadamard_qwen35::recognizes_model_type,
        discover_prism_hadamard_qwen35_model,
    ),
    ModelFamilyDescription::recognized_by_model_type(
        MODERNBERT_MODEL_FAMILY,
        ModelKind::Embedding,
        modernbert::recognizes_model_type,
        discover_modernbert_model,
    ),
    ModelFamilyDescription::recognized_by_model_type(
        LAYA_MODEL_FAMILY,
        ModelKind::Decision,
        laya::recognizes_model_type,
        discover_laya_model,
    )
    .with_bundled_models(discover_laya_bundled_checkpoints),
];

/// The registered description whose family has this identifier.
pub(super) fn registered_description_named(
    identifier: &str,
) -> Option<&'static ModelFamilyDescription> {
    REGISTERED_MODEL_FAMILIES
        .iter()
        .find(|registered_description| registered_description.model_family.as_str() == identifier)
}

#[cfg(test)]
#[path = "model_family_registry_tests.rs"]
mod model_family_registry_tests;
