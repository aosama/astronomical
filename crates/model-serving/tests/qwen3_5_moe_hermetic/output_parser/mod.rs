use astronomical_ipc_protocol::ChatToolDefinition;
use astronomical_model_serving::{
    Qwen3_5MoEOutputEvent, Qwen3_5MoEOutputParser, Qwen3_5MoEToolCall,
};
use serde_json::Value;

const THINK_START: &str = concat!("<", "think", ">");
const THINK_END: &str = concat!("<", "/think", ">");
const TOOL_CALL_START: &str = concat!("<", "tool_call", ">");
const TOOL_CALL_END: &str = concat!("<", "/tool_call", ">");

mod model_correction;
mod permissive_arguments;
mod reasoning;
mod schema_forms;
mod tool_call_lifecycle;
