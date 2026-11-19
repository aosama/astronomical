//! Qwen3.5 output-parser unit contracts, beside the parser they pin.
//!
//! Happy-path, marker-defect, foreign-dialect, schema, and streaming groups
//! cover the shared tool-call contract. The parser is fail-open by design:
//! malformed or foreign tool-call syntax reaches the harness as a tool call or
//! visible text, never an abort. Fixtures use the Romeo and Juliet literary
//! tools per the workspace test-fixture mandate.

use astronomical_ipc_protocol::ChatToolDefinition;

use super::Qwen3_5OutputParser;

mod foreign_dialect_tests;
mod happy_path_permutations_tests;
mod marker_permutations_tests;
mod permissive_arguments_tests;
mod prose_marker_hijack_tests;
mod reasoning_tests;
mod schema_forms_tests;
mod tool_call_lifecycle_tests;
mod undeclared_function_tests;

const THINK_START: &str = concat!("<", "think", ">");
const THINK_END: &str = concat!("<", "/think", ">");
const TOOL_CALL_START: &str = concat!("<", "tool_call", ">");
const TOOL_CALL_END: &str = concat!("<", "/tool_call", ">");

// Re-exported under short names for the foreign-dialect suite.
const TC_OPEN: &str = TOOL_CALL_START;
const TC_CLOSE: &str = TOOL_CALL_END;

const DECLARED_CHARACTER_FUNCTION: &str = "find_character";
const DECLARED_SCENE_FUNCTION: &str = "summarize_scene";
const UNDECLARED_FUNCTION_NAME: &str = "inspect_verse";
const CHARACTER_NAME: &str = "Romeo";
const ROMEO_ARGUMENTS_JSON: &str = r#"{"name":"Romeo"}"#;
const BALCONY_ARGUMENTS_JSON: &str = r#"{"scene":"balcony"}"#;
const EMPTY_ARGUMENTS_JSON: &str = "{}";

fn literary_declared_tools() -> Vec<ChatToolDefinition> {
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

fn literary_output_parser() -> Qwen3_5OutputParser {
    Qwen3_5OutputParser::new(&literary_declared_tools())
        .expect("Romeo and Juliet literary tools should construct a Qwen3.5 parser")
}
