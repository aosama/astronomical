use std::path::PathBuf;

use astronomical_daemon::{
    ChatGenerationStreamEvent, GenerationStartError, build_application,
    build_application_with_discovered_models,
};
use axum::{
    body::{Body, to_bytes},
    http::{Request, StatusCode, header},
};
use config::DiscoveredModel;
use ipc_protocol::{
    ChatGenerationCompletionReason, ChatGenerationFailureReason, MAX_IPC_FRAME_BYTES,
};
use serde_json::Value;
use tower::ServiceExt;

use super::scripted_chat_executor::{SCRIPTED_CHAT_MODEL_ID as MODEL_ID, ScriptedChatExecutor};

mod request_rejection;
mod streaming;
mod success;
mod support;

use support::*;
