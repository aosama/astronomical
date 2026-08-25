//! Owns config-gated, attributed loading of the optional Qwen `thinking.md` seed.
//!
//! Missing, empty, unreadable, or non-UTF-8 files are absent. The supervisor never
//! creates this file; the user authors it when they want a thinking-channel seed.

use std::{path::Path, time::Duration};

use astronomical_ipc_protocol::MAX_QWEN_THINKING_CHANNEL_SEED_BYTES;
use tokio::io::AsyncReadExt;

use crate::{
    SupervisorPerformanceMeasurement, SupervisorPerformanceOperation, application::ApplicationState,
};

const QWEN_THINKING_CHANNEL_SEED_LOAD_TIMEOUT: Duration = Duration::from_secs(2);

pub(crate) async fn load_configured_qwen_thinking_channel_seed(
    application_state: &ApplicationState,
    model_id: &str,
) -> Option<String> {
    let should_load_seed = application_state
        .reloadable_config
        .as_ref()
        .and_then(|reloadable_config| reloadable_config.read().ok())
        .is_some_and(|resolved_runtime_config| {
            resolved_runtime_config.experimental_qwen_thinking_channel_seed_enabled
                && resolved_runtime_config
                    .discovered_models
                    .iter()
                    .any(|model| {
                        model.model_id == model_id
                            && model.model_family == astronomical_config::ModelFamily::Qwen3_5
                    })
        });
    if !should_load_seed {
        return None;
    }
    let thinking_channel_seed_file_path = application_state
        .runtime_config_resolver
        .as_ref()?
        .instance_paths()
        .qwen_thinking_channel_seed_file_path();
    application_state
        .supervisor_attribution_log
        .measure_async_operation_best_effort(
            SupervisorPerformanceOperation::QwenThinkingChannelSeedLoad,
            || load_qwen_thinking_channel_seed_outcome(true, &thinking_channel_seed_file_path),
            |load_outcome| {
                if load_outcome.is_success() {
                    SupervisorPerformanceMeasurement::success()
                } else {
                    SupervisorPerformanceMeasurement::failure()
                }
            },
        )
        .await
        .into_seed()
}

/// Reads a bounded Qwen thinking-channel seed from instance state when enabled.
#[must_use]
pub async fn load_qwen_thinking_channel_seed(
    thinking_channel_seed_enabled: bool,
    thinking_channel_seed_file_path: &Path,
) -> Option<String> {
    load_qwen_thinking_channel_seed_outcome(
        thinking_channel_seed_enabled,
        thinking_channel_seed_file_path,
    )
    .await
    .into_seed()
}

async fn load_qwen_thinking_channel_seed_outcome(
    thinking_channel_seed_enabled: bool,
    thinking_channel_seed_file_path: &Path,
) -> QwenThinkingChannelSeedLoadOutcome {
    if !thinking_channel_seed_enabled {
        return QwenThinkingChannelSeedLoadOutcome::Absent;
    }
    match tokio::time::timeout(
        QWEN_THINKING_CHANNEL_SEED_LOAD_TIMEOUT,
        read_qwen_thinking_channel_seed_outcome(thinking_channel_seed_file_path),
    )
    .await
    {
        Ok(load_outcome) => load_outcome,
        Err(_elapsed) => {
            tracing::debug!(
                timeout_milliseconds = %QWEN_THINKING_CHANNEL_SEED_LOAD_TIMEOUT.as_millis(),
                "ignored a Qwen thinking-channel seed file that exceeded its read timeout"
            );
            QwenThinkingChannelSeedLoadOutcome::Failed
        }
    }
}

async fn read_qwen_thinking_channel_seed_outcome(
    thinking_channel_seed_file_path: &Path,
) -> QwenThinkingChannelSeedLoadOutcome {
    let thinking_seed_file = match tokio::fs::File::open(thinking_channel_seed_file_path).await {
        Ok(thinking_seed_file) => thinking_seed_file,
        Err(io_error) if io_error.kind() == std::io::ErrorKind::NotFound => {
            return QwenThinkingChannelSeedLoadOutcome::Absent;
        }
        Err(io_error) => {
            tracing::debug!(
                error = %io_error,
                "ignored an unreadable Qwen thinking-channel seed file"
            );
            return QwenThinkingChannelSeedLoadOutcome::Failed;
        }
    };
    let mut bounded_thinking_seed_file = thinking_seed_file.take(
        u64::try_from(MAX_QWEN_THINKING_CHANNEL_SEED_BYTES)
            .unwrap_or(u64::MAX)
            .saturating_add(1),
    );
    let mut file_contents = String::new();
    match bounded_thinking_seed_file
        .read_to_string(&mut file_contents)
        .await
    {
        Ok(file_size_bytes) if file_size_bytes <= MAX_QWEN_THINKING_CHANNEL_SEED_BYTES => {
            let trimmed_thinking_markdown = file_contents.trim();
            if trimmed_thinking_markdown.is_empty() {
                return QwenThinkingChannelSeedLoadOutcome::Absent;
            }
            QwenThinkingChannelSeedLoadOutcome::Loaded(trimmed_thinking_markdown.to_owned())
        }
        Ok(file_size_bytes) => {
            tracing::debug!(
                file_size_bytes,
                maximum_file_size_bytes = MAX_QWEN_THINKING_CHANNEL_SEED_BYTES,
                "ignored an oversized Qwen thinking-channel seed file"
            );
            QwenThinkingChannelSeedLoadOutcome::Failed
        }
        Err(io_error) => {
            tracing::debug!(
                error = %io_error,
                "ignored an unreadable Qwen thinking-channel seed file"
            );
            QwenThinkingChannelSeedLoadOutcome::Failed
        }
    }
}

enum QwenThinkingChannelSeedLoadOutcome {
    Loaded(String),
    Absent,
    Failed,
}

impl QwenThinkingChannelSeedLoadOutcome {
    const fn is_success(&self) -> bool {
        !matches!(self, Self::Failed)
    }

    fn into_seed(self) -> Option<String> {
        match self {
            Self::Loaded(thinking_channel_seed) => Some(thinking_channel_seed),
            Self::Absent | Self::Failed => None,
        }
    }
}
