//! How discovery recognizes the ModernBERT embedding family (issue #2473, Step 5): the
//! family's recognizers, its folder checks and its description, which the
//! family list in `model-family-registry` names. It lives in the family's
//! own folder, beside the engine, and like discovery it never links MLX,
//! because the daemon recognizes folders through it.

mod family_description;
mod modernbert;

pub use family_description::{MODERNBERT_FAMILY_DESCRIPTION, discover_modernbert_model};
pub use modernbert::MODERNBERT_MODEL_FAMILY;
