//! The conditions a model was served under, read from the engine's own status
//! after it has loaded the model: the memory limit it applied, whether the
//! weights streamed from the SSD, and the drafter in use.
//!
//! They come from the engine and not from the command line, because a figure
//! is only comparable across releases when its conditions are what the engine
//! really did: a requested limit the engine adjusted, or a drafter it refused
//! to load, would otherwise be recorded as if it had happened.

use serde_json::Value;

use crate::release_speed::speed_ledger::{DrafterInUse, WeightsPlacement};

#[derive(Clone, Debug, PartialEq)]
pub struct ServedConditions {
    pub maximum_mlx_memory_gb: Option<u64>,
    pub mlx_memory_limit_bytes: u64,
    pub weights_placement: WeightsPlacement,
    pub drafter: DrafterInUse,
    pub multi_token_prediction_draft_tokens: Option<u32>,
}

/// The conditions of the model the engine is serving, which must be the one
/// just measured.
pub fn served_conditions(
    status_document: &Value,
    model_id: &str,
) -> Result<ServedConditions, String> {
    let ready_model_id = status_document["ready_model_id"].as_str();
    if ready_model_id != Some(model_id) {
        return Err(format!(
            "the engine reports {} ready, not {model_id}",
            ready_model_id.unwrap_or("no model")
        ));
    }
    let mlx_memory_limit_bytes = status_document["mlx_memory_ceiling_bytes"]
        .as_u64()
        .ok_or("the engine's status gives no memory limit")?;
    Ok(ServedConditions {
        maximum_mlx_memory_gb: status_document["configured_maximum_mlx_memory_gb"].as_u64(),
        mlx_memory_limit_bytes,
        weights_placement: weights_placement(status_document, mlx_memory_limit_bytes),
        drafter: drafter_in_use(&status_document["draft_model"]),
        multi_token_prediction_draft_tokens: status_document["configuration"]["ready_model"]
            ["chunking"]["multi_token_prediction_draft_tokens"]["effective"]
            .as_u64()
            .and_then(|draft_tokens| u32::try_from(draft_tokens).ok()),
    })
}

/// The engine reports how the loaded model's weights are held: `resident`
/// when all of them are in memory (a dense model reports it too), `hybrid` or
/// `paged` when some stream from the SSD. An engine that reports no mode is
/// judged by the model's size against the limit, which is when the engine
/// would serve it from the SSD.
fn weights_placement(status_document: &Value, mlx_memory_limit_bytes: u64) -> WeightsPlacement {
    match status_document["expert_memory_mode"].as_str() {
        Some("resident") => WeightsPlacement::InMemory,
        Some(_) => WeightsPlacement::Streamed,
        None if status_document["ready_model_size_bytes"]
            .as_u64()
            .is_some_and(|model_size_bytes| model_size_bytes > mlx_memory_limit_bytes) =>
        {
            WeightsPlacement::Streamed
        }
        None => WeightsPlacement::InMemory,
    }
}

/// Only a drafter the engine loaded and drafts with counts; one that is
/// installed but unavailable under this load is none.
fn drafter_in_use(draft_model_status: &Value) -> DrafterInUse {
    if draft_model_status["state"] != "drafting" {
        return DrafterInUse::None;
    }
    match draft_model_status["generation"].as_str() {
        Some("dflash") => DrafterInUse::Dflash,
        Some("dflash2") => DrafterInUse::Dflash2,
        _ => DrafterInUse::None,
    }
}

#[cfg(test)]
#[path = "served_conditions_tests.rs"]
mod served_conditions_tests;
