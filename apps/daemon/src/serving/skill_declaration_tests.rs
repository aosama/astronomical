use std::{
    path::{Path, PathBuf},
    sync::{Arc, RwLock},
};

use ipc_protocol::ChatMessage;

use super::{declare_astronomical_skill, skill_declaration_text};
use crate::{ResolvedRuntimeConfig, SkillSettings};

const SKILL_INDEX_PATH: &str = "/state/skill/astronomical/SKILL.md";

fn reloadable_config(declared_to_models: bool) -> Arc<RwLock<ResolvedRuntimeConfig>> {
    let mut resolved_config = crate::config::config_reload_tests::sample_resolved_config();
    resolved_config.skill_settings = SkillSettings {
        declared_to_models,
        updates_from_web: false,
        index_path: PathBuf::from(SKILL_INDEX_PATH),
    };
    Arc::new(RwLock::new(resolved_config))
}

fn user_message(user_text: &str) -> ChatMessage {
    ChatMessage::User {
        content: user_text.to_owned(),
        images: Vec::new(),
    }
}

fn system_text(chat_messages: &[ChatMessage]) -> &str {
    match chat_messages.first() {
        Some(ChatMessage::System { content }) => content,
        _ => panic!("the first message is not a system prompt"),
    }
}

#[test]
fn should_append_the_declaration_after_the_harness_own_system_prompt() {
    let harness_system_prompt = "You are a coding agent.";
    let mut chat_messages = vec![
        ChatMessage::System {
            content: harness_system_prompt.to_owned(),
        },
        user_message("Draw me a lighthouse."),
    ];
    declare_astronomical_skill(Some(&reloadable_config(true)), &mut chat_messages);
    assert_eq!(chat_messages.len(), 2);
    let declared_system_text = system_text(&chat_messages);
    assert!(declared_system_text.starts_with(harness_system_prompt));
    assert!(declared_system_text.ends_with(&skill_declaration_text(Path::new(SKILL_INDEX_PATH))));
}

#[test]
fn should_create_a_system_prompt_for_a_request_that_has_none() {
    let mut chat_messages = vec![user_message("Is Astronomical running?")];
    declare_astronomical_skill(Some(&reloadable_config(true)), &mut chat_messages);
    assert_eq!(chat_messages.len(), 2);
    assert_eq!(
        system_text(&chat_messages),
        skill_declaration_text(Path::new(SKILL_INDEX_PATH))
    );
    assert!(matches!(chat_messages[1], ChatMessage::User { .. }));
}

#[test]
fn should_leave_the_request_untouched_when_switched_off() {
    let mut chat_messages = vec![user_message("Explain this regex.")];
    let untouched_messages = chat_messages.clone();
    declare_astronomical_skill(Some(&reloadable_config(false)), &mut chat_messages);
    assert_eq!(chat_messages, untouched_messages);
}

#[test]
fn should_leave_the_request_untouched_without_a_resolved_configuration() {
    let mut chat_messages = vec![user_message("Explain this regex.")];
    let untouched_messages = chat_messages.clone();
    declare_astronomical_skill(None, &mut chat_messages);
    assert_eq!(chat_messages, untouched_messages);
}

/// The same request must produce the same prompt bytes every time, or the
/// `astro_ssd_prompt_cache` could never reuse a conversation's beginning.
#[test]
fn should_produce_identical_text_for_identical_requests() {
    let mut first_messages = vec![user_message("Hello")];
    let mut second_messages = first_messages.clone();
    let shared_config = reloadable_config(true);
    declare_astronomical_skill(Some(&shared_config), &mut first_messages);
    declare_astronomical_skill(Some(&shared_config), &mut second_messages);
    assert_eq!(first_messages, second_messages);
}

#[test]
fn should_name_the_abilities_the_file_and_the_way_out() {
    let declaration_text = skill_declaration_text(Path::new(SKILL_INDEX_PATH));
    for expected_phrase in [
        "draw images",
        "whether it is running",
        "text aloud",
        "describe pictures",
        SKILL_INDEX_PATH,
        "If you have no way to read files, ignore this note.",
    ] {
        assert!(
            declaration_text.contains(expected_phrase),
            "missing {expected_phrase}"
        );
    }
}
