//! `POST /v1/messages` driven through the whole application router with a
//! scripted worker (issue #890): response shapes, the SSE event sequence an
//! Anthropic client parses, and Anthropic-shaped rejections.

use astronomical_daemon::{ChatGenerationStreamEvent, GenerationStartError, build_application};
use astronomical_ipc_protocol::{ChatGenerationCompletionReason, ChatGenerationFailureReason};
use axum::{
    body::{Body, to_bytes},
    http::{Request, StatusCode, header},
};
use serde_json::Value;
use tower::ServiceExt;

use super::openai_responses_endpoint::MODEL_ID;
use super::openai_responses_endpoint::support::ScriptedResponsesExecutor;

mod request_rejection;
mod streaming;
mod success;
mod support;

use support::*;
