//! How discovery recognizes the Kokoro speech family (issue #2473, Step 5): the
//! family's recognizers, its folder checks and its description, which the
//! family list in `model-family-registry` names. It lives in the family's
//! own folder, beside the engine, and like discovery it never links MLX,
//! because the daemon recognizes folders through it.

mod family_description;
mod kokoro;

pub use family_description::{KOKORO_FAMILY_DESCRIPTION, discover_kokoro_model};
pub use kokoro::KOKORO_MODEL_FAMILY;
pub use kokoro::{
    KOKORO_CONFIGURATION_FILE_NAME, KOKORO_VOICES_DIRECTORY_NAME, KOKORO_WEIGHTS_FILE_NAME,
    is_kokoro_configuration, kokoro_speech_capabilities,
};
