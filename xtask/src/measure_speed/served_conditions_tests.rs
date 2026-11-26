use serde_json::json;

use super::{ServedConditions, last_reply_peak_mlx_memory_bytes, served_conditions};
use crate::release_speed::speed_ledger::{DrafterInUse, WeightsPlacement};

const SIXTEEN_GB_BYTES: u64 = 16 * 1_073_741_824;

fn status_serving(model_id: &str) -> serde_json::Value {
    json!({
        "ready_model_id": model_id,
        "ready_model_size_bytes": 20 * 1_073_741_824_u64,
        "mlx_memory_ceiling_bytes": SIXTEEN_GB_BYTES,
        "configured_maximum_mlx_memory_gb": 16,
        "expert_memory_mode": "paged",
        "draft_model": {"state": "drafting", "generation": "dflash", "location": "bundled", "tree_node_budget": 3},
        "configuration": {"ready_model": {"chunking": {"multi_token_prediction_draft_tokens": {
            "is_configured": false, "configured": null, "default": 1, "effective": 1
        }}}}
    })
}

fn placement_of(status_document: &serde_json::Value) -> WeightsPlacement {
    served_conditions(status_document, "Model-A")
        .expect("a served model")
        .weights_placement
}

fn drafter_of(status_document: &serde_json::Value) -> DrafterInUse {
    served_conditions(status_document, "Model-A")
        .expect("a served model")
        .drafter
}

#[test]
fn should_read_a_streamed_model_drafting_with_dflash_under_a_limit() {
    let conditions = served_conditions(&status_serving("Model-A"), "Model-A")
        .expect("the engine serves Model-A");

    assert_eq!(
        conditions,
        ServedConditions {
            maximum_mlx_memory_gb: Some(16),
            mlx_memory_limit_bytes: SIXTEEN_GB_BYTES,
            weights_placement: WeightsPlacement::Streamed,
            drafter: DrafterInUse::Dflash,
            multi_token_prediction_draft_tokens: Some(1),
        }
    );
}

#[test]
fn should_refuse_when_the_engine_serves_another_model_or_gives_no_limit() {
    assert!(served_conditions(&status_serving("Model-B"), "Model-A").is_err());

    let mut status_without_limit = status_serving("Model-A");
    status_without_limit["mlx_memory_ceiling_bytes"] = json!(null);
    assert!(served_conditions(&status_without_limit, "Model-A").is_err());
}

#[test]
fn should_hold_resident_experts_in_memory_and_size_a_dense_model_against_the_limit() {
    let mut resident = status_serving("Model-A");
    resident["expert_memory_mode"] = json!("resident");
    let mut dense_larger_than_limit = status_serving("Model-A");
    dense_larger_than_limit["expert_memory_mode"] = json!(null);
    let mut dense_within_limit = dense_larger_than_limit.clone();
    dense_within_limit["ready_model_size_bytes"] = json!(SIXTEEN_GB_BYTES);

    assert_eq!(placement_of(&resident), WeightsPlacement::InMemory);
    assert_eq!(
        placement_of(&dense_larger_than_limit),
        WeightsPlacement::Streamed
    );
    assert_eq!(
        placement_of(&dense_within_limit),
        WeightsPlacement::InMemory,
        "a model exactly the size of the limit fits"
    );
}

#[test]
fn should_count_only_a_drafter_that_drafts() {
    let mut unavailable = status_serving("Model-A");
    unavailable["draft_model"] = json!({"state": "unavailable", "reason": "not enough memory"});
    let mut second_generation = status_serving("Model-A");
    second_generation["draft_model"]["generation"] = json!("dflash2");
    let mut older_engine = status_serving("Model-A");
    older_engine["draft_model"] = json!(null);
    older_engine["configuration"] = json!(null);

    assert_eq!(drafter_of(&unavailable), DrafterInUse::None);
    assert_eq!(drafter_of(&second_generation), DrafterInUse::Dflash2);
    assert_eq!(drafter_of(&older_engine), DrafterInUse::None);
    assert_eq!(
        served_conditions(&older_engine, "Model-A")
            .expect("an older engine's status still reads")
            .multi_token_prediction_draft_tokens,
        None
    );
}

#[test]
fn should_read_the_last_replys_peak_memory_and_none_from_an_engine_that_reports_none() {
    let reporting_status =
        serde_json::json!({"last_reply_peak_mlx_memory_bytes": 19_327_352_832_u64});
    let silent_status = serde_json::json!({"last_reply_peak_mlx_memory_bytes": null});

    assert_eq!(
        last_reply_peak_mlx_memory_bytes(&reporting_status),
        Some(19_327_352_832)
    );
    assert_eq!(last_reply_peak_mlx_memory_bytes(&silent_status), None);
    assert_eq!(
        last_reply_peak_mlx_memory_bytes(&serde_json::json!({})),
        None
    );
}
