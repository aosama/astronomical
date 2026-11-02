//! One line of a calibration mix and how it renders to text.

use serde::Deserialize;

/// One chat turn in a chat-style sample.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
pub struct CalibrationMessage {
    pub role: String,
    pub content: String,
}

/// One raw line of the mix.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct CalibrationSample {
    #[serde(default)]
    pub domain: String,
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub messages: Option<Vec<CalibrationMessage>>,
    /// Tool definitions, carried as opaque JSON; the fallback renderer does
    /// not inline them.
    #[serde(default)]
    pub tools: Option<serde_json::Value>,
}

impl CalibrationSample {
    /// The text to encode: the role-prefixed rendering of `messages`
    /// (`[role]: content` joined by blank lines), else `text` as-is. Messages
    /// win when a sample carries both, as in OptiQ. Samples with neither are
    /// skipped.
    #[must_use]
    pub fn render_text(&self) -> Option<String> {
        if let Some(messages) = &self.messages {
            let mut rendered_turns = Vec::with_capacity(messages.len());
            for message in messages {
                rendered_turns.push(format!("[{}]: {}", message.role, message.content));
            }
            return Some(rendered_turns.join("\n\n"));
        }
        self.text.clone()
    }
}
