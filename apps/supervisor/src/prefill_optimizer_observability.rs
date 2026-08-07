use astronomical_config::PrefillChunckSizingPolicy;
use std::sync::{Arc, RwLock};

use astronomical_ipc_protocol::WorkerPrefillOptimizerInsight;
use serde_json::{Value, json};

use crate::WorkerHealthSnapshot;

const MAXIMUM_RECENT_PREFILL_OPTIMIZER_INSIGHTS: usize = 12;

pub(crate) fn record_prefill_optimizer_insight(
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    prefill_optimizer_insight: WorkerPrefillOptimizerInsight,
) {
    let Ok(mut worker_health_snapshot) = health_snapshot.write() else {
        return;
    };
    if worker_health_snapshot.prefill_optimizer_insights.len()
        >= MAXIMUM_RECENT_PREFILL_OPTIMIZER_INSIGHTS
    {
        worker_health_snapshot.prefill_optimizer_insights.remove(0);
    }
    worker_health_snapshot
        .prefill_optimizer_insights
        .push(prefill_optimizer_insight);
}

pub(crate) fn prefill_optimizer_status_document(
    prefill_chunck_sizing_policy: Option<&PrefillChunckSizingPolicy>,
    recent_prefill_optimizer_insights: &[WorkerPrefillOptimizerInsight],
) -> Value {
    let latest_prefill_optimizer_insight = recent_prefill_optimizer_insights.last();
    let (optimizer_is_enabled, candidate_prefill_chunck_tokens, fixed_prefill_chunck_tokens) =
        match prefill_chunck_sizing_policy {
            Some(PrefillChunckSizingPolicy::Optimized {
                optimizer_prefill_chunck_token_candidates,
            }) => (
                Some(true),
                optimizer_prefill_chunck_token_candidates.clone(),
                None,
            ),
            Some(PrefillChunckSizingPolicy::Fixed {
                fixed_prefill_chunck_tokens,
            }) => (Some(false), Vec::new(), Some(*fixed_prefill_chunck_tokens)),
            None => (
                latest_prefill_optimizer_insight.map(|_| true),
                latest_prefill_optimizer_insight.map_or_else(Vec::new, |latest_insight| {
                    latest_insight
                        .candidate_evidence
                        .iter()
                        .map(|candidate_evidence| {
                            candidate_evidence.candidate_prefill_chunck_tokens
                        })
                        .collect()
                }),
                None,
            ),
        };
    let recent_transitions = recent_prefill_optimizer_insights
        .iter()
        .map(|prefill_optimizer_insight| {
            json!({
                "requested_prefill_chunck_tokens": prefill_optimizer_insight.requested_prefill_chunck_tokens,
                "actual_prefill_chunck_tokens": prefill_optimizer_insight.actual_prefill_chunck_tokens,
                "elapsed_millis": prefill_optimizer_insight.elapsed_millis,
                "decision_reason": prefill_optimizer_insight.decision_reason,
                "has_observed_prefill_capacity_constraint": prefill_optimizer_insight.has_observed_prefill_capacity_constraint,
                "context": prefill_optimizer_insight.context,
            })
        })
        .collect::<Vec<_>>();
    json!({
        "enabled": optimizer_is_enabled,
        "candidate_prefill_chunck_tokens": candidate_prefill_chunck_tokens,
        "fixed_prefill_chunck_tokens": fixed_prefill_chunck_tokens,
        "latest_insight": latest_prefill_optimizer_insight,
        "recent_transitions": recent_transitions,
    })
}
