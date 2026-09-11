//! Asks for a Library chat model only when the session has several and the
//! user did not already pass `--model`. Non-TTY sessions never prompt.

use std::io::{BufRead, Write};

use crate::{errors::LaunchError, models::LibraryChatModel};

const OPENCODE_PREFERRED_CONTEXT_WINDOW_TOKENS: u32 = 65_536;

pub fn select_chat_model(
    chat_models: &[LibraryChatModel],
    requested_model_id: Option<&str>,
    is_interactive: bool,
    stdin: &mut dyn BufRead,
    stderr: &mut dyn Write,
) -> Result<LibraryChatModel, LaunchError> {
    if chat_models.is_empty() {
        return Err(LaunchError::NoChatModels);
    }
    if let Some(requested_model_id) = requested_model_id {
        return chat_models
            .iter()
            .find(|chat_model| chat_model.model_id == requested_model_id)
            .cloned()
            .ok_or_else(|| LaunchError::RequestedModelMissing {
                requested_model_id: requested_model_id.to_owned(),
            });
    }
    if chat_models.len() == 1 {
        return Ok(chat_models[0].clone());
    }
    if !is_interactive {
        return Err(LaunchError::ModelPickerRequired);
    }
    writeln!(stderr, "Select a model:").map_err(|_| LaunchError::InvalidModelSelection)?;
    for (model_index, chat_model) in chat_models.iter().enumerate() {
        writeln!(stderr, "{}. {}", model_index + 1, chat_model.model_id)
            .map_err(|_| LaunchError::InvalidModelSelection)?;
    }
    let mut selected_line = String::new();
    stdin
        .read_line(&mut selected_line)
        .map_err(|_| LaunchError::InvalidModelSelection)?;
    let selected_text = selected_line.trim();
    if selected_text.is_empty() {
        return Err(LaunchError::InvalidModelSelection);
    }
    if let Ok(selected_number) = selected_text.parse::<usize>() {
        if selected_number >= 1 {
            if let Some(chat_model) = chat_models.get(selected_number - 1) {
                return Ok(chat_model.clone());
            }
        }
        return Err(LaunchError::InvalidModelSelection);
    }
    chat_models
        .iter()
        .find(|chat_model| chat_model.model_id == selected_text)
        .cloned()
        .ok_or(LaunchError::InvalidModelSelection)
}

pub fn warn_if_context_window_is_narrow(chat_model: &LibraryChatModel, stderr: &mut dyn Write) {
    let Some(context_window_tokens) = chat_model.context_window_tokens else {
        return;
    };
    if context_window_tokens >= OPENCODE_PREFERRED_CONTEXT_WINDOW_TOKENS {
        return;
    }
    // Advisory only: a write failure must not block launch.
    let _ = writeln!(
        stderr,
        "OpenCode works better with a 64k or larger context window; {} advertises {context_window_tokens} tokens.",
        chat_model.model_id
    );
}
