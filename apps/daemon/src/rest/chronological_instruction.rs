//! Chronological placement of mid-conversation instructions, shared by every
//! surface whose clients send system turns after the conversation starts.
//! The chat template has one leading system slot, so later instructions ride
//! on the user turn they follow, inside an escaped `<system-update>` tag the
//! model cannot confuse with user text.

use astronomical_ipc_protocol::ChatMessage;

const CHRONOLOGICAL_INSTRUCTION_OPENING_TAG: &str = "<system-update>\n";
const CHRONOLOGICAL_INSTRUCTION_CLOSING_TAG: &str = "\n</system-update>";

/// Places one mid-conversation instruction chronologically: in the system
/// prompt when no conversation turn precedes it, and otherwise inside an
/// escaped `<system-update>` tag on the latest user message. Both the
/// Responses and the Anthropic Messages surfaces receive mid-conversation
/// system turns, and both place them here.
pub(crate) fn append_instruction(chat_messages: &mut Vec<ChatMessage>, instruction_content: &str) {
    if place_before_conversation(chat_messages, instruction_content) {
        return;
    }
    if let Some(ChatMessage::User {
        content: prior_user_content,
        ..
    }) = chat_messages.last_mut()
    {
        prior_user_content.push('\n');
        append_escaped_instruction(prior_user_content, instruction_content);
        return;
    }
    let mut chronological_instruction = String::new();
    append_escaped_instruction(&mut chronological_instruction, instruction_content);
    chat_messages.push(ChatMessage::User {
        content: chronological_instruction,
        images: Vec::new(),
    });
}

/// Puts an instruction that precedes every conversation turn into the system
/// prompt, creating it or extending it, and reports whether it did. Extending
/// rather than opening a synthetic user turn keeps words out of the user's
/// mouth before they spoke.
fn place_before_conversation(
    chat_messages: &mut Vec<ChatMessage>,
    instruction_content: &str,
) -> bool {
    if chat_messages.is_empty() {
        chat_messages.push(ChatMessage::System {
            content: instruction_content.to_owned(),
        });
        return true;
    }
    if let [
        ChatMessage::System {
            content: system_content,
        },
    ] = chat_messages.as_mut_slice()
    {
        system_content.push_str("\n\n");
        system_content.push_str(instruction_content);
        return true;
    }
    false
}

fn append_escaped_instruction(target_content: &mut String, instruction_content: &str) {
    target_content.push_str(CHRONOLOGICAL_INSTRUCTION_OPENING_TAG);
    for instruction_character in instruction_content.chars() {
        match instruction_character {
            '&' => target_content.push_str("&amp;"),
            '<' => target_content.push_str("&lt;"),
            '>' => target_content.push_str("&gt;"),
            _ => target_content.push(instruction_character),
        }
    }
    target_content.push_str(CHRONOLOGICAL_INSTRUCTION_CLOSING_TAG);
}

#[cfg(test)]
mod tests {
    use astronomical_ipc_protocol::ChatMessage;

    use super::append_instruction;

    fn system_message(content: &str) -> ChatMessage {
        ChatMessage::System {
            content: content.to_owned(),
        }
    }

    fn user_message(content: &str) -> ChatMessage {
        ChatMessage::User {
            content: content.to_owned(),
            images: Vec::new(),
        }
    }

    #[test]
    fn should_become_the_system_prompt_when_nothing_precedes_it() {
        let mut chat_messages = Vec::new();

        append_instruction(&mut chat_messages, "Speak as Prospero.");

        assert_eq!(chat_messages, vec![system_message("Speak as Prospero.")]);
    }

    #[test]
    fn should_extend_a_lone_system_prompt_instead_of_inventing_a_user_turn() {
        let mut chat_messages = vec![system_message("Speak as Prospero.")];

        append_instruction(&mut chat_messages, "Answer in verse.");

        assert_eq!(
            chat_messages,
            vec![system_message("Speak as Prospero.\n\nAnswer in verse.")]
        );
    }

    #[test]
    fn should_tag_and_escape_an_instruction_after_a_user_turn() {
        let mut chat_messages = vec![user_message("Read Act 4.")];

        append_instruction(&mut chat_messages, "Prefer <scene 1> & no other.");

        assert_eq!(
            chat_messages,
            vec![user_message(
                "Read Act 4.\n<system-update>\nPrefer &lt;scene 1&gt; &amp; no other.\n</system-update>"
            )]
        );
    }

    #[test]
    fn should_open_a_tagged_user_turn_after_an_assistant_turn() {
        let mut chat_messages = vec![
            user_message("Read Act 4."),
            ChatMessage::Assistant {
                content: Some("Our revels now are ended.".to_owned()),
                reasoning_content: None,
                tool_calls: Vec::new(),
            },
        ];

        append_instruction(&mut chat_messages, "Stop here.");

        assert_eq!(
            chat_messages[2],
            user_message("<system-update>\nStop here.\n</system-update>")
        );
    }
}
