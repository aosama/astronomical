use crate::{ChatGenerationSettings, ChatMessage};

// The sampling values sit well inside every validation limit so a test that
// varies one field cannot trip an unrelated range check.
pub(crate) fn standard_settings() -> ChatGenerationSettings {
    ChatGenerationSettings {
        max_output_tokens: 512,
        temperature_thousandths: Some(600),
        top_p_thousandths: Some(950),
        repetition_penalty_thousandths: None,
        seed: Some(7),
        thinking_budget: None,
    }
}

pub(crate) fn user_messages() -> Vec<ChatMessage> {
    vec![ChatMessage::User {
        content: "Inspect the repository.".to_owned(),
        images: Vec::new(),
    }]
}
