//! How discovery recognizes the Qwen 3.5 and Prism Hadamard chat families (issue #2473, Step 5): the
//! family's recognizers, its folder checks and its description, which the
//! family list in `model-family-registry` names. It lives in the family's
//! own folder, beside the engine, and like discovery it never links MLX,
//! because the daemon recognizes folders through it.

mod family_description;
mod prism_hadamard_qwen35;
mod qwen3_5;

pub use family_description::{
    PRISM_HADAMARD_QWEN35_FAMILY_DESCRIPTION, QWEN3_5_FAMILY_DESCRIPTION,
    discover_prism_hadamard_qwen35_model, discover_qwen3_5_model,
};
pub use prism_hadamard_qwen35::PRISM_HADAMARD_QWEN35_MODEL_FAMILY;
pub use qwen3_5::QWEN3_5_MODEL_FAMILY;
pub use qwen3_5::{
    MINIMUM_SERVABLE_CONTEXT_WINDOW_TOKENS, context_window_tokens, required_shard_file_names,
};
