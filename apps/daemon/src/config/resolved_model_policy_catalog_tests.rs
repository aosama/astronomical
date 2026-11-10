//! Unit contracts for how a discovered decision model is named to the worker.

use astronomical_config::ModelFamily;
use astronomical_ipc_protocol::WorkerDecisionModelFamily;

use super::worker_decision_family;

#[test]
fn should_name_a_kev_model_kev_to_the_worker() {
    assert_eq!(
        worker_decision_family(ModelFamily::Kev),
        WorkerDecisionModelFamily::Kev
    );
}

#[test]
fn should_name_a_clef_model_clef_to_the_worker() {
    assert_eq!(
        worker_decision_family(ModelFamily::Clef),
        WorkerDecisionModelFamily::Clef
    );
}

#[test]
fn should_name_a_laya_model_laya_to_the_worker() {
    assert_eq!(
        worker_decision_family(ModelFamily::Laya),
        WorkerDecisionModelFamily::Laya
    );
}

#[test]
fn should_name_a_family_without_a_decision_engine_laya_for_the_worker_to_refuse() {
    assert_eq!(
        worker_decision_family(ModelFamily::Qwen3_5),
        WorkerDecisionModelFamily::Laya,
        "only Kev and Laya folders report a decision capability; anything else is refused at load"
    );
}
