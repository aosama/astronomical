//! The stock label and the reply sentence every Observatory surface and the menu share.
//!
//! Stock describes where the expert library sits right now, from the seated
//! byte counts, or for a dense model on the streaming engine, how much of
//! its decoder layers it reads from the SSD (issue #1985). It is a short label rather than a full sentence: the menu
//! shows it in a half-width monospaced cell, so every stock label stays
//! within MAXIMUM_STOCK_LABEL_CHARACTERS to fit on one line there. The reply
//! sentence describes what the current or last reply did with storage, from
//! the request-scoped memory mode. Clients print these strings. They do not
//! keep a second phrase table.

use astronomical_ipc_protocol::{ExpertMemoryMode, WorkerDenseLayerResidency};

use crate::{ExpertResidencySnapshot, WorkerActivity, WorkerHealthSnapshot};

const APPLE_GIGABYTE_BYTES: f64 = 1_073_741_824.0;

/// What a dense model says instead of a sentence about experts it does not
/// have, when it reports no decoder layers: the resident engine serves only
/// a dense model that fits whole, and says so in these words, where the
/// streaming engine, which may stream, counts its layers instead.
const DENSE_MODEL_REPLY_SENTENCE: &str = "Dense model, no experts.";

/// Copy budget for the stock label, in characters. The menu's Residency cell
/// is the tightest place the label appears, and 18 characters leaves margin
/// there. Layer counts up to 999 stay within it.
pub(crate) const MAXIMUM_STOCK_LABEL_CHARACTERS: usize = 18;

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
    let stock_label = stock_sentence(ready_model_id, expert_memory_mode, expert_residency);
    // Guards future copy edits in debug builds; the unit test also enforces this limit.
    debug_assert!(
        stock_label.chars().count() <= MAXIMUM_STOCK_LABEL_CHARACTERS,
        "stock label {stock_label:?} would wrap in the menu's Residency cell"
    );
    ExpertPlacementCopy {
        stock_sentence: stock_label,
        reply_sentence: reply_sentence(
            ready_model_id,
            activity,
            expert_memory_mode,
            expert_residency,
        ),
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
    if let Some(dense_layers) = expert_residency.dense_layers {
        residency_json["dense_layers"] = serde_json::json!({
            "decoder_layer_count": dense_layers.decoder_layer_count,
            "kept_layer_count": dense_layers.kept_layer_count,
            "streamed_layer_payload_bytes": dense_layers.streamed_layer_payload_bytes,
        });
    }
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
        return "No model loaded".to_owned();
    }
    if let Some(dense_layers) = dense_layers_of(expert_residency) {
        return dense_stock_sentence(dense_layers);
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
        return "All on SSD".to_owned();
    }
    on_ssd_stock_sentence(
        complete_expert_payload_byte_count.saturating_sub(resident_expert_payload_bytes),
    )
}

/// The one wording for bytes a model reads from the SSD, sparse or dense.
fn on_ssd_stock_sentence(on_ssd_bytes: u64) -> String {
    format!(
        "{:.2} GB on SSD",
        on_ssd_bytes as f64 / APPLE_GIGABYTE_BYTES
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
        Some(ExpertMemoryMode::Hybrid | ExpertMemoryMode::Paged) => "Size not measured",
        Some(ExpertMemoryMode::Resident) | None => "All in RAM",
    }
}

fn fully_seated_stock_sentence(total_layer_count: u32) -> String {
    if total_layer_count > 0 {
        return format!("{total_layer_count} layers in RAM");
    }
    "All in RAM".to_owned()
}

fn reply_sentence(
    ready_model_id: Option<&str>,
    activity: WorkerActivity,
    expert_memory_mode: Option<ExpertMemoryMode>,
    expert_residency: Option<ExpertResidencySnapshot>,
) -> String {
    // A mode left over from a worker that has no ready model must not talk
    // about a reply. There is no library to have read.
    if ready_model_id_is_absent(ready_model_id) {
        return String::new();
    }
    if let Some(dense_layers) = dense_layers_of(expert_residency) {
        return dense_reply_sentence(dense_layers);
    }
    let Some(expert_memory_mode) = expert_memory_mode else {
        return String::new();
    };
    if is_dense_model_residency(expert_residency) {
        return DENSE_MODEL_REPLY_SENTENCE.to_owned();
    }
    if reply_is_running(activity) {
        return running_reply_sentence(expert_memory_mode).to_owned();
    }
    idle_reply_sentence(expert_memory_mode).to_owned()
}

/// The worker builds an expert pager only for a mixture-of-experts model, and
/// without one it reports no layers, no expert capacity and no expert bytes.
/// All three must be zero: a capacity of zero alone can also mean a sparse
/// model whose plans did not name an expert count. Without a measurement the
/// model's kind is unknown and the mode sentence stands.
fn is_dense_model_residency(expert_residency: Option<ExpertResidencySnapshot>) -> bool {
    expert_residency.is_some_and(|measured_residency| {
        measured_residency.expert_capacity == 0
            && measured_residency.total_layer_count == 0
            && measured_residency.complete_expert_payload_byte_count == 0
    })
}

fn dense_layers_of(
    expert_residency: Option<ExpertResidencySnapshot>,
) -> Option<WorkerDenseLayerResidency> {
    // A report of no layers names nothing to place, so the model's dense
    // wording without counts stands.
    expert_residency
        .and_then(|measured_residency| measured_residency.dense_layers)
        .filter(|dense_layers| dense_layers.decoder_layer_count > 0)
}

/// A dense model that streams says how much it reads from the SSD, the way
/// a sparse model does; one that keeps every layer says so.
fn dense_stock_sentence(dense_layers: WorkerDenseLayerResidency) -> String {
    if dense_layers.streamed_layer_payload_bytes == 0 {
        return fully_seated_stock_sentence(dense_layers.decoder_layer_count);
    }
    on_ssd_stock_sentence(dense_layers.streamed_layer_payload_bytes)
}

/// Names the kept and streamed layers. A streamed layer is read from the SSD
/// for every forward that runs it.
fn dense_reply_sentence(dense_layers: WorkerDenseLayerResidency) -> String {
    let WorkerDenseLayerResidency {
        decoder_layer_count,
        kept_layer_count,
        ..
    } = dense_layers;
    if kept_layer_count >= decoder_layer_count {
        return format!("Dense model: all {decoder_layer_count} layers in RAM.");
    }
    if kept_layer_count == 0 {
        return format!("Dense model: all {decoder_layer_count} layers read from the SSD.");
    }
    format!(
        "Dense model: {kept_layer_count} of {decoder_layer_count} layers in RAM, {} read from the SSD.",
        decoder_layer_count - kept_layer_count
    )
}

fn reply_is_running(activity: WorkerActivity) -> bool {
    match activity {
        WorkerActivity::PromptProcessing
        | WorkerActivity::GenerationPreparation
        | WorkerActivity::Generating => true,
        WorkerActivity::Idle | WorkerActivity::ImageGeneration | WorkerActivity::Speaking => false,
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
