//! `POST /v1/messages` and `POST /v1/messages/count_tokens` driven through
//! the whole application router with a scripted worker (issue #890):
//! response shapes, the SSE event sequence an Anthropic client parses, token
//! counts, and Anthropic-shaped rejections.

use astronomical_daemon::{ChatGenerationStreamEvent, GenerationStartError, build_application};
use astronomical_ipc_protocol::{ChatGenerationCompletionReason, ChatGenerationFailureReason};
use axum::{
    body::{Body, to_bytes},
    http::{Request, StatusCode, header},
};
use serde_json::Value;
use tower::ServiceExt;

use super::scripted_chat_executor::{SCRIPTED_CHAT_MODEL_ID as MODEL_ID, ScriptedChatExecutor};

mod count_tokens;
mod generation_failures;
mod request_rejection;
mod streaming;
mod success;
mod support;

use support::*;
