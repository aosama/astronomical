//! Every model family's description, in the order a folder is classified
//! (issue #2473, Step 5).
//!
//! One explicit list, rather than registration at link time, so a reader
//! sees every family in one place and a family left out fails the registry
//! tests instead of vanishing at run time. Within each way of recognizing a
//! folder (pipeline index, marker file, `model_type`) the first family that
//! claims it wins, so the order below is the classification order. Each
//! family declares its description in its own recognition crate, in the
//! family's folder.

use model_discovery::ModelFamilyDescription;

/// One line per family; its description lives in the family's own folder.
pub(crate) static REGISTERED_MODEL_FAMILIES: [ModelFamilyDescription; 10] = [
    flux2_klein_recognition::FLUX2_KLEIN_FAMILY_DESCRIPTION,
    qwen_image_21_recognition::QWEN_IMAGE_21_FAMILY_DESCRIPTION,
    kev_recognition::KEV_FAMILY_DESCRIPTION,
    clef_recognition::CLEF_FAMILY_DESCRIPTION,
    kokoro_recognition::KOKORO_FAMILY_DESCRIPTION,
    phonon_recognition::PHONON_FAMILY_DESCRIPTION,
    qwen35_recognition::QWEN3_5_FAMILY_DESCRIPTION,
    qwen35_recognition::PRISM_HADAMARD_QWEN35_FAMILY_DESCRIPTION,
    modernbert_recognition::MODERNBERT_FAMILY_DESCRIPTION,
    laya_recognition::LAYA_FAMILY_DESCRIPTION,
];

#[cfg(test)]
#[path = "registered_model_families_tests.rs"]
mod registered_model_families_tests;
