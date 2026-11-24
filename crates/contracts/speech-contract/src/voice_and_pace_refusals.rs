//! The words a caller reads when a voice or a pace is refused. The REST
//! endpoint refuses them from what discovery knows before a model loads, and
//! the engine refuses them again after it loads; both say the same thing.

/// A model that cannot change its pace refuses any speed but 1.0.
pub const NATURAL_PACE_ONLY_REASON: &str =
    "this voice model speaks only at its natural pace (speed 1.0)";

/// Names the voice asked for and the voices the model has.
pub fn unknown_voice_reason(voice_name: &str, voice_names: &[String]) -> String {
    format!(
        "there is no voice {voice_name:?}; the voices are {}",
        voice_names.join(", ")
    )
}

#[cfg(test)]
#[path = "voice_and_pace_refusals_tests.rs"]
mod tests;
