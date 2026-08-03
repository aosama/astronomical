use std::{future::Future, path::PathBuf, pin::Pin};

use astronomical_config::DiscoveredModel;
use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationCompletionReason, ChatGenerationFailureReason,
    ChatModelCapabilities, MtpRuntimeState,
};
use astronomical_supervisor::{
    ChatGenerationExecutor, ChatGenerationStreamEvent, GenerationStartError, WorkerHealthSnapshot,
    build_application, build_application_with_config_warning_and_discovered_models,
};
use axum::{
    body::{Body, to_bytes},
    http::{Request, StatusCode, header},
};
use serde_json::Value;
use tokio::sync::mpsc;
use tower::ServiceExt;

const MODEL_ID: &str = "astronomical/responses-endpoint-test-model";

mod request_rejection;
mod streaming;
mod success;
mod support;

use support::*;
