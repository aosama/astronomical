//! Qwen3.5 dense and MoE share this output parser. Happy-path, marker-defect,
//! foreign-dialect, schema, and streaming tests cover that shared tool-call
//! contract. The parser is fail-open by design: malformed or foreign tool-call
//! syntax reaches the harness as a tool call or visible text, never an abort.

use astronomical_ipc_protocol::ChatToolDefinition;
use astronomical_model_serving::{Qwen3_5OutputEvent, Qwen3_5OutputParser, Qwen3_5ToolCall};
use serde_json::Value;

const THINK_START: &str = concat!("<", "think", ">");
const THINK_END: &str = concat!("<", "/think", ">");
const TOOL_CALL_START: &str = concat!("<", "tool_call", ">");
const TOOL_CALL_END: &str = concat!("<", "/tool_call", ">");

// Re-exported under short names for the foreign-dialect suite.
const TC_OPEN: &str = TOOL_CALL_START;
const TC_CLOSE: &str = TOOL_CALL_END;

mod foreign_dialect;
mod happy_path_permutations;
mod marker_permutations;
mod permissive_arguments;
mod reasoning;
mod schema_forms;
mod support;
mod tool_call_lifecycle;
mod undeclared_function;
