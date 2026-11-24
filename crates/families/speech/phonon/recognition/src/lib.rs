//! How discovery recognizes the Phonon speech family (issue #2473, Step 5): the
//! family's recognizers, its folder checks and its description, which the
//! family list in `model-family-registry` names. It lives in the family's
//! own folder, beside the engine, and like discovery it never links MLX,
//! because the daemon recognizes folders through it.

mod family_description;
mod phonon;

pub use family_description::{PHONON_FAMILY_DESCRIPTION, discover_phonon_model};
pub use phonon::PHONON_MODEL_FAMILY;
pub use phonon::{
    PHONON_CONFIGURATION_FILE_NAME, PHONON_DEFAULT_VOICE_FILE_NAME, PHONON_DEFAULT_VOICE_NAME,
    PHONON_TOKENIZER_FILE_NAME, PHONON_VOICES_DIRECTORY_NAME, PHONON_WEIGHTS_FILE_NAME,
    is_phonon_configuration, phonon_speech_capabilities,
};
