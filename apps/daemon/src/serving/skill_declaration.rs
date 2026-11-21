//! Tells the model a request reaches that the Astronomical skill exists.
//!
//! Astronomical ships an inference engine, not an agentic harness, so it
//! cannot put a skill in any harness's skill list. The engine is the one
//! layer behind every harness: it appends a short declaration to the system
//! prompt, and the model reads the skill's file with the harness's own file
//! tool when a task calls for it (issue #2397).
//!
//! Every chat request gets it, with or without tool definitions: some
//! harnesses describe their tools in the prompt text instead of the request,
//! and the engine cannot tell those from a plain chat.

use std::{
    path::Path,
    sync::{Arc, RwLock},
};

use ipc_protocol::ChatMessage;

use crate::ResolvedRuntimeConfig;

/// The declaration for a skill whose index file is at `skill_index_path`.
///
/// The wording was settled by measurement. A declaration that named only the
/// product was skipped for "draw me a picture", in two harnesses, while one
/// naming the abilities a person asks for was followed. Then, with the
/// harness's provider itself named "astronomical", "is Astronomical running?"
/// was answered from the provider list without the skill until the questions
/// about Astronomical were listed first, with examples. The last sentence
/// keeps a model without any way to read files from pretending it did.
pub(crate) fn skill_declaration_text(skill_index_path: &Path) -> String {
    format!(
        "This Mac runs Astronomical, which serves this model and can also draw images, read \
         text aloud, describe pictures and run decision models, all locally with nothing \
         leaving the Mac. Its skill explains how. Read the skill before you answer any question \
         about Astronomical (whether it is running, which models it has, why it is slow, how to \
         set it up or connect a tool), and before you draw an image, produce speech, describe a \
         picture or run an `astronomical` command; prefer it over other image or speech tools.\n\n\
         Read: {}\n\
         If you have no way to read files, ignore this note.",
        skill_index_path.display()
    )
}

/// Appends the declaration to the request's system prompt when the person's
/// settings ask for it, creating the system prompt when there is none.
///
/// It goes at the end so a harness's own opening text stays byte-identical
/// from request to request, which is what lets the `astro_ssd_prompt_cache`
/// reuse its work on the shared beginning of a conversation.
pub(crate) fn declare_astronomical_skill(
    reloadable_runtime_config: Option<&Arc<RwLock<ResolvedRuntimeConfig>>>,
    chat_messages: &mut Vec<ChatMessage>,
) {
    let Some(reloadable_runtime_config) = reloadable_runtime_config else {
        return;
    };
    let Ok(resolved_runtime_config) = reloadable_runtime_config.read() else {
        return;
    };
    let skill_settings = &resolved_runtime_config.skill_settings;
    if !skill_settings.declared_to_models {
        return;
    }
    append_to_system_prompt(
        chat_messages,
        &skill_declaration_text(&skill_settings.index_path),
    );
}

fn append_to_system_prompt(chat_messages: &mut Vec<ChatMessage>, declaration_text: &str) {
    match chat_messages.first_mut() {
        Some(ChatMessage::System { content }) => {
            content.push_str("\n\n");
            content.push_str(declaration_text);
        }
        _ => chat_messages.insert(
            0,
            ChatMessage::System {
                content: declaration_text.to_owned(),
            },
        ),
    }
}

#[cfg(test)]
#[path = "skill_declaration_tests.rs"]
mod skill_declaration_tests;
