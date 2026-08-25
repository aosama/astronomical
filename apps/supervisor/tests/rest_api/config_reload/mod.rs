//! Supervisor control endpoint tests: `POST /v1/config/reload` and
//! `POST /v1/control/shutdown`.

use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use astronomical_supervisor::{
    ChatGenerationExecutor, GenerationPerformanceLog, ResolvedRuntimeConfig,
    ResolvedRuntimeConfigResolver, ShutdownController, WorkerHandle, WorkerHealthStatus,
    build_application_with_full_control, build_application_with_shutdown,
    build_development_application_with_reload,
};
use axum::{
    body::{Body, to_bytes},
    http::{Request, StatusCode, header},
};
use tokio::time::{Instant, sleep, timeout};
use tower::ServiceExt;

use crate::common::ScriptedExecutor;

mod configuration_status;
mod generation_admission;
mod maximum_mlx_memory;
mod mixed_reload_configuration_generation;
mod output_limits;
mod qwen_thinking_channel_seed;
mod reload_status;
mod reload_validation;
mod shutdown;
mod support;
mod transactional_replacement;

use support::*;
