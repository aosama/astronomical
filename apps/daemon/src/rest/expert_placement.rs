//! The two sentences every Observatory surface and the menu share.
//!
//! Stock describes where the expert library sits right now, from the seated
//! byte counts. The reply sentence describes what the current or last reply
//! did with storage, from the request-scoped memory mode. Clients print these
//! strings. They do not keep a second phrase table.

use astronomical_ipc_protocol::ExpertMemoryMode;

use crate::{ExpertResidencySnapshot, WorkerActivity, WorkerHealthSnapshot};

const APPLE_GIGABYTE_BYTES: f64 = 1_073_741_824.0;

pub(crate) struct ExpertPlacementCopy {
    pub(crate) stock_sentence: String,
    pub(crate) reply_sentence: String,
}

pub(crate) fn expert_placement(
    ready_model_id: Option<&str>,
    activity: WorkerActivity,
    expert_memory_mode: Option<ExpertMemoryMode>,
    expert_residency: Option<ExpertResidencySnapshot>,
) -> ExpertPlacementCopy {
    ExpertPlacementCopy {
        stock_sentence: stock_sentence(ready_model_id, expert_memory_mode, expert_residency),
        reply_sentence: reply_sentence(ready_model_id, activity, expert_memory_mode),
    }
}

/// Attaches the shared sentences and the residency document, including the
/// live layer counts when this sample has a map.
pub(crate) fn attach_expert_placement_and_residency(
    status_json: &mut serde_json::Value,
    worker_health_snapshot: &WorkerHealthSnapshot,
) {
    let placement = expert_placement(
        worker_health_snapshot.ready_model_id.as_deref(),
        worker_health_snapshot.activity,
        worker_health_snapshot.expert_memory_mode,
        worker_health_snapshot.expert_residency,
    );
    status_json["expert_placement"] = serde_json::json!({
        "stock_sentence": placement.stock_sentence,
        "reply_sentence": placement.reply_sentence,
    });
    status_json["expert_residency"] =
        expert_residency_json(worker_health_snapshot.expert_residency);
}

pub(crate) fn expert_residency_json(
    expert_residency: Option<ExpertResidencySnapshot>,
) -> serde_json::Value {
    let Some(expert_residency) = expert_residency else {
        return serde_json::Value::Null;
    };
    let mut residency_json = serde_json::json!({
        "total_layer_count": expert_residency.total_layer_count,
        "resident_expert_count": expert_residency.resident_expert_count,
        "resident_expert_payload_bytes": expert_residency.resident_expert_payload_bytes,
        "complete_expert_payload_byte_count": expert_residency.complete_expert_payload_byte_count,
        "expert_capacity": expert_residency.expert_capacity,
    });
    let retained_expert_counts = expert_residency.layer_seating.published_counts();
    if !retained_expert_counts.is_empty() {
        residency_json["retained_expert_counts"] = serde_json::json!(retained_expert_counts);
    }
    residency_json
}

fn stock_sentence(
    ready_model_id: Option<&str>,
    expert_memory_mode: Option<ExpertMemoryMode>,
    expert_residency: Option<ExpertResidencySnapshot>,
) -> String {
    if ready_model_id_is_absent(ready_model_id) {
        return "No expert library is loaded.".to_owned();
    }
    let (complete_expert_payload_byte_count, resident_expert_payload_bytes, total_layer_count) =
        residency_bytes(expert_residency);
    if complete_expert_payload_byte_count == 0 {
        return unmeasured_stock_sentence(expert_memory_mode).to_owned();
    }
    if resident_expert_payload_bytes >= complete_expert_payload_byte_count {
        return fully_seated_stock_sentence(total_layer_count);
    }
    if resident_expert_payload_bytes == 0 {
        return "The expert library is still on the SSD.".to_owned();
    }
    format!(
        "{:.2} GB are still on the SSD.",
        left_on_ssd_gigabytes(
            complete_expert_payload_byte_count,
            resident_expert_payload_bytes
        )
    )
}

fn ready_model_id_is_absent(ready_model_id: Option<&str>) -> bool {
    match ready_model_id {
        None => true,
        Some(model_id) => model_id.is_empty(),
    }
}

fn residency_bytes(expert_residency: Option<ExpertResidencySnapshot>) -> (u64, u64, u32) {
    match expert_residency {
        Some(expert_residency) => (
            expert_residency.complete_expert_payload_byte_count,
            expert_residency.resident_expert_payload_bytes,
            expert_residency.total_layer_count,
        ),
        None => (0, 0, 0),
    }
}

fn unmeasured_stock_sentence(expert_memory_mode: Option<ExpertMemoryMode>) -> &'static str {
    match expert_memory_mode {
        Some(ExpertMemoryMode::Hybrid | ExpertMemoryMode::Paged) => {
            "Expert library size is not measured yet."
        }
        Some(ExpertMemoryMode::Resident) | None => "Fully in memory.",
    }
}

fn fully_seated_stock_sentence(total_layer_count: u32) -> String {
    if total_layer_count > 0 {
        return format!("All {total_layer_count} layers are in RAM.");
    }
    "The expert library is in RAM.".to_owned()
}

fn left_on_ssd_gigabytes(
    complete_expert_payload_byte_count: u64,
    resident_expert_payload_bytes: u64,
) -> f64 {
    let left_on_ssd_bytes =
        complete_expert_payload_byte_count.saturating_sub(resident_expert_payload_bytes);
    left_on_ssd_bytes as f64 / APPLE_GIGABYTE_BYTES
}

fn reply_sentence(
    ready_model_id: Option<&str>,
    activity: WorkerActivity,
    expert_memory_mode: Option<ExpertMemoryMode>,
) -> String {
    // A mode left over from a worker that has no ready model must not talk
    // about a reply. There is no library to have read.
    if ready_model_id_is_absent(ready_model_id) {
        return String::new();
    }
    let Some(expert_memory_mode) = expert_memory_mode else {
        return String::new();
    };
    if reply_is_running(activity) {
        return running_reply_sentence(expert_memory_mode).to_owned();
    }
    idle_reply_sentence(expert_memory_mode).to_owned()
}

fn reply_is_running(activity: WorkerActivity) -> bool {
    match activity {
        WorkerActivity::PromptProcessing
        | WorkerActivity::GenerationPreparation
        | WorkerActivity::Generating => true,
        WorkerActivity::Idle | WorkerActivity::ImageGeneration => false,
    }
}

fn running_reply_sentence(expert_memory_mode: ExpertMemoryMode) -> &'static str {
    match expert_memory_mode {
        ExpertMemoryMode::Resident => "This reply is reading experts from RAM.",
        ExpertMemoryMode::Hybrid => "This reply has read experts from the SSD.",
        ExpertMemoryMode::Paged => "This reply is streaming experts from the SSD.",
    }
}

fn idle_reply_sentence(expert_memory_mode: ExpertMemoryMode) -> &'static str {
    match expert_memory_mode {
        ExpertMemoryMode::Resident => "Experts are being served from RAM.",
        ExpertMemoryMode::Hybrid => "The last reply read experts from the SSD.",
        ExpertMemoryMode::Paged => "Experts stream from the SSD when a reply needs them.",
    }
}
