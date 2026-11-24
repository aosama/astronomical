//! How discovery recognizes the Kev decision family (issue #2473, Step 5): the
//! family's recognizers, its folder checks and its description, which the
//! family list in `model-family-registry` names. It lives in the family's
//! own folder, beside the engine, and like discovery it never links MLX,
//! because the daemon recognizes folders through it.

mod family_description;
mod kev;

pub use family_description::{KEV_FAMILY_DESCRIPTION, discover_kev_model};
pub use kev::KEV_MODEL_FAMILY;
