//! How discovery recognizes the Laya decision family (issue #2473, Step 5): the
//! family's recognizers, its folder checks and its description, which the
//! family list in `model-family-registry` names. It lives in the family's
//! own folder, beside the engine, and like discovery it never links MLX,
//! because the daemon recognizes folders through it.

mod family_description;
mod laya;

pub use family_description::{
    LAYA_FAMILY_DESCRIPTION, discover_laya_bundled_checkpoints, discover_laya_model,
};
pub use laya::LAYA_MODEL_FAMILY;
