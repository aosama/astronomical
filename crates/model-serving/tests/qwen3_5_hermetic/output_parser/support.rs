use astronomical_ipc_protocol::ChatToolDefinition;
use astronomical_model_serving::Qwen3_5OutputParser;

pub(super) const DECLARED_CHARACTER_FUNCTION: &str = "find_character";
pub(super) const DECLARED_SCENE_FUNCTION: &str = "summarize_scene";
pub(super) const UNDECLARED_FUNCTION_NAME: &str = "inspect_verse";
pub(super) const CHARACTER_NAME: &str = "Romeo";
pub(super) const ROMEO_ARGUMENTS_JSON: &str = r#"{"name":"Romeo"}"#;
pub(super) const BALCONY_ARGUMENTS_JSON: &str = r#"{"scene":"balcony"}"#;
pub(super) const EMPTY_ARGUMENTS_JSON: &str = "{}";

pub(super) fn literary_declared_tools() -> Vec<ChatToolDefinition> {
    vec![
        ChatToolDefinition {
            name: DECLARED_CHARACTER_FUNCTION.to_owned(),
            description: Some("Locate a character in Romeo and Juliet.".to_owned()),
            parameters_json:
                r#"{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}"#
                    .to_owned(),
        },
        ChatToolDefinition {
            name: DECLARED_SCENE_FUNCTION.to_owned(),
            description: Some("Summarize a scene in Romeo and Juliet.".to_owned()),
            parameters_json:
                r#"{"type":"object","properties":{"scene":{"type":"string"}},"required":["scene"]}"#
                    .to_owned(),
        },
    ]
}

pub(super) fn literary_output_parser() -> Qwen3_5OutputParser {
    Qwen3_5OutputParser::new(&literary_declared_tools())
        .expect("Romeo and Juliet literary tools should construct a Qwen3.5 parser")
}
