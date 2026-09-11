//! Issue #510 chat-lane parity: journeys read the same unused-ceiling split
//! the menu shows, and they keep that evidence after the run.
//!
//! Image generation and embeddings are a later slice: those engines do not yet
//! compose this vocabulary.

use std::{fs, net::SocketAddr, path::Path};

use serde_json::Value;
use tokio::time::{Duration, Instant, sleep};

use super::serving_rest::get_json_endpoint;

const NAMED_HEADROOM_FIELDS: [&str; 4] = [
    "reserved_model_core_slack_bytes",
    "reserved_context_growth_bytes",
    "reserved_activation_and_workspace_bytes",
    "unseated_expert_entitlement_bytes",
];

/// Fails when the status snapshot is missing the split or the identity does
/// not close. Call this on load-complete, restore, prefill, decode, and idle
/// samples — the same documents the menu paints.
pub(crate) fn assert_status_memory_ceiling_utilization_closes(status_document: &Value) {
    let utilization = &status_document["mlx_memory_snapshot"]["memory_ceiling_utilization"];
    assert!(
        utilization.is_object(),
        "the status snapshot must publish memory_ceiling_utilization so journeys can read the unused split the menu shows: {status_document}"
    );
    let unused_headroom_bytes = utilization["unused_headroom_bytes"]
        .as_u64()
        .expect("unused_headroom_bytes must be present on the published split");
    let named_headroom_bytes = NAMED_HEADROOM_FIELDS
        .into_iter()
        .map(|field_name| utilization[field_name].as_u64().unwrap_or(0))
        .sum::<u64>();
    let speculative_draft_payload_bytes = utilization["speculative_draft_payload_bytes"]
        .as_u64()
        .unwrap_or(0);
    let unexplained_headroom_bytes = utilization["unexplained_headroom_bytes"]
        .as_u64()
        .unwrap_or(u64::MAX);
    let owner_overrun_bytes = utilization["owner_overrun_bytes"]
        .as_u64()
        .unwrap_or(u64::MAX);
    assert_eq!(
        unexplained_headroom_bytes, 0,
        "the status-published decomposition must close with no residual: {utilization}"
    );
    assert_eq!(
        owner_overrun_bytes, 0,
        "the status-published decomposition must report no owner overrun: {utilization}"
    );
    let explained_headroom_bytes =
        named_headroom_bytes.saturating_sub(speculative_draft_payload_bytes);
    assert!(
        explained_headroom_bytes <= unused_headroom_bytes,
        "named owners must not overrun unused headroom: unused={unused_headroom_bytes} explained={explained_headroom_bytes} utilization={utilization}"
    );
}

/// Writes the status split next to worker logs so a run can be compared later.
pub(crate) fn preserve_status_memory_ceiling_utilization(
    journey_evidence_name: &str,
    isolated_worker_home: &Path,
    status_document: &Value,
) {
    let unix_millis = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|unix_time| unix_time.as_millis())
        .unwrap_or_default();
    let evidence_directory = std::env::current_dir()
        .expect("the journey should resolve its working directory")
        .join("target/acceptance-evidence")
        .join(journey_evidence_name)
        .join(unix_millis.to_string());
    fs::create_dir_all(&evidence_directory)
        .expect("the acceptance evidence directory should be created");
    let evidence_document = serde_json::json!({
        "status_memory_ceiling_utilization": status_document["mlx_memory_snapshot"]["memory_ceiling_utilization"],
        "status_mlx_memory_snapshot": status_document["mlx_memory_snapshot"],
        "activity": status_document["activity"],
        "snapshot_source": status_document["mlx_memory_snapshot"]["source"],
    });
    fs::write(
        evidence_directory.join("memory-utilization.json"),
        serde_json::to_string_pretty(&evidence_document).expect("evidence should serialize"),
    )
    .expect("the utilization evidence should be written");
    let logging_directory = isolated_worker_home.join(".astronomical-dev").join("logs");
    let Ok(logging_entries) = fs::read_dir(&logging_directory) else {
        return;
    };
    for logging_entry in logging_entries.flatten() {
        let source_path = logging_entry.path();
        if source_path.is_file()
            && let Some(source_name) = source_path.file_name()
        {
            let _ = fs::copy(&source_path, evidence_directory.join(source_name));
        }
    }
}

pub(crate) fn assert_and_preserve_status_memory_ceiling_utilization(
    journey_evidence_name: &str,
    isolated_worker_home: &Path,
    status_document: &Value,
) {
    assert_status_memory_ceiling_utilization_closes(status_document);
    preserve_status_memory_ceiling_utilization(
        journey_evidence_name,
        isolated_worker_home,
        status_document,
    );
}

/// Ready can land a moment before the load-complete memory sample. The menu
/// polls; journeys must too, or they assert a status the user never stares at.
pub(crate) async fn wait_for_status_memory_ceiling_utilization(
    server_address: SocketAddr,
) -> Value {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let status_document = get_json_endpoint(server_address, "/v1/status").await;
        if status_document["mlx_memory_snapshot"]["memory_ceiling_utilization"].is_object() {
            assert_status_memory_ceiling_utilization_closes(&status_document);
            return status_document;
        }
        assert!(
            Instant::now() < deadline,
            "status never published memory_ceiling_utilization after load: {status_document}"
        );
        sleep(Duration::from_millis(50)).await;
    }
}
