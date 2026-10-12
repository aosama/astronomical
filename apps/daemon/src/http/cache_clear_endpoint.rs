use std::path::{Component, Path};

use axum::{
    Json,
    extract::{Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde::{Deserialize, Serialize};

use crate::PromptCacheClearOutcome;
use crate::application::ApplicationState;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CacheClearQuery {
    model: Option<String>,
}

#[derive(Serialize)]
struct CacheClearResponse {
    status: &'static str,
    model_id: Option<String>,
    blocks_removed: u64,
    bytes_freed: u64,
}

/// Deletes global or model-scoped SSD prompt-cache content through the worker.
pub(crate) async fn clear_cache(
    State(application_state): State<ApplicationState>,
    Query(cache_clear_query): Query<CacheClearQuery>,
) -> Response {
    let Some(worker_handle) = application_state.worker_control.as_ref() else {
        return (StatusCode::NOT_FOUND, "live worker control is unavailable").into_response();
    };
    if let Some(model_id) = cache_clear_query.model.as_deref()
        && !is_safe_model_id(model_id)
    {
        return (
            StatusCode::BAD_REQUEST,
            "model must be a safe relative model ID",
        )
            .into_response();
    }
    match worker_handle
        .clear_prompt_cache(cache_clear_query.model.clone())
        .await
    {
        Ok(PromptCacheClearOutcome::Applied {
            model_id,
            blocks_removed,
            bytes_freed,
        }) => (
            StatusCode::OK,
            Json(CacheClearResponse {
                status: "cleared",
                model_id,
                blocks_removed,
                bytes_freed,
            }),
        )
            .into_response(),
        Ok(PromptCacheClearOutcome::Queued) => (
            StatusCode::ACCEPTED,
            Json(CacheClearResponse {
                status: "queued",
                model_id: cache_clear_query.model,
                blocks_removed: 0,
                bytes_freed: 0,
            }),
        )
            .into_response(),
        Err(worker_control_error) => {
            tracing::error!(error = %worker_control_error, "prompt-cache clear failed");
            (StatusCode::SERVICE_UNAVAILABLE, "worker cache clear failed").into_response()
        }
    }
}

fn is_safe_model_id(model_id: &str) -> bool {
    !model_id.is_empty()
        && !model_id.contains('\0')
        && !model_id.contains('\\')
        && Path::new(model_id)
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
}
