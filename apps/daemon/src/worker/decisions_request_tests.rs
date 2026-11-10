//! Unit tests for the decisions capability gate.

use std::sync::{Arc, RwLock};

use astronomical_ipc_protocol::{
    MINIMUM_DECISION_TOKENS_PER_QUESTION, WorkerDecisionCapabilities, WorkerEmbeddingCapabilities,
    WorkerModelCapabilities,
};

use super::resident_model_answers_decisions;
use crate::WorkerHealthSnapshot;

fn health_snapshot_with(
    ready_model_capabilities: Option<WorkerModelCapabilities>,
) -> Arc<RwLock<WorkerHealthSnapshot>> {
    let Some(ready_model_capabilities) = ready_model_capabilities else {
        return Arc::new(RwLock::new(WorkerHealthSnapshot::ready_without_model(
            8 * 1024 * 1024 * 1024,
        )));
    };
    Arc::new(RwLock::new(WorkerHealthSnapshot::ready_with_model(
        String::from("resident-model"),
        ready_model_capabilities,
    )))
}

#[test]
fn should_accept_a_resident_decision_model() {
    let health_snapshot = health_snapshot_with(Some(WorkerModelCapabilities::decisions(
        WorkerDecisionCapabilities {
            maximum_tokens_per_question: 512,
        },
    )));
    assert!(resident_model_answers_decisions(&health_snapshot));
}

#[test]
fn should_accept_the_smallest_usable_token_limit_and_reject_one_below() {
    let smallest_usable = health_snapshot_with(Some(WorkerModelCapabilities::decisions(
        WorkerDecisionCapabilities {
            maximum_tokens_per_question: MINIMUM_DECISION_TOKENS_PER_QUESTION,
        },
    )));
    let one_below = health_snapshot_with(Some(WorkerModelCapabilities::decisions(
        WorkerDecisionCapabilities {
            maximum_tokens_per_question: MINIMUM_DECISION_TOKENS_PER_QUESTION - 1,
        },
    )));
    assert!(resident_model_answers_decisions(&smallest_usable));
    assert!(!resident_model_answers_decisions(&one_below));
}

#[test]
fn should_reject_a_resident_model_of_another_kind() {
    let health_snapshot = health_snapshot_with(Some(WorkerModelCapabilities::embeddings(
        WorkerEmbeddingCapabilities {
            vector_width: 768,
            max_input_tokens: 8_192,
        },
    )));
    assert!(!resident_model_answers_decisions(&health_snapshot));
}

#[test]
fn should_reject_when_no_model_is_resident() {
    assert!(!resident_model_answers_decisions(&health_snapshot_with(
        None
    )));
}
