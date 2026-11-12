//! Which names a voice may have.
//!
//! A voice is a file in a model's voices folder, named after the voice. A
//! name of letters, digits, hyphens and underscores only can never reach
//! outside that folder, so the same rule decides which files discovery lists
//! as voices and which names a request may ask for: a voice that is
//! advertised can always be loaded.

pub fn is_plain_voice_name(voice_name: &str) -> bool {
    !voice_name.is_empty()
        && voice_name.chars().all(|character| {
            character.is_ascii_alphanumeric() || character == '_' || character == '-'
        })
}

#[cfg(test)]
#[path = "voice_name_tests.rs"]
mod tests;
