//! How discovery recognizes the FLUX.2 Klein image family (issue #2473, Step 5): the
//! family's recognizers, its folder checks and its description, which the
//! family list in `model-family-registry` names. It lives in the family's
//! own folder, beside the engine, and like discovery it never links MLX,
//! because the daemon recognizes folders through it.

mod family_description;
mod flux2_klein;
mod flux2_klein_documents;
mod flux2_klein_mlx_4bit;

pub use family_description::{FLUX2_KLEIN_FAMILY_DESCRIPTION, discover_flux2_klein_model};
pub use flux2_klein::FLUX2_KLEIN_MODEL_FAMILY;
pub use flux2_klein::{
    Flux2KleinDirectoryVerificationError,
    verify_model_directory as verify_flux2_klein_model_directory,
};
pub use flux2_klein_mlx_4bit::{
    FLUX2_KLEIN_MLX_4BIT_TRANSFORMER_INDEX_FILE_NAME, classify_mflux_transformer_index_bytes,
};
