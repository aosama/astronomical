//! How discovery recognizes the Qwen-Image-2.1 image family (issue #2473, Step 5): the
//! family's recognizers, its folder checks and its description, which the
//! family list in `model-family-registry` names. It lives in the family's
//! own folder, beside the engine, and like discovery it never links MLX,
//! because the daemon recognizes folders through it.

mod family_description;
mod qwen_image_21;
mod qwen_image_21_documents;

pub use family_description::{QWEN_IMAGE_21_FAMILY_DESCRIPTION, discover_qwen_image_21_model};
pub use qwen_image_21::QWEN_IMAGE_21_MODEL_FAMILY;
pub use qwen_image_21::{
    QwenImage21DirectoryVerificationError,
    verify_model_directory as verify_qwen_image_21_model_directory,
};
