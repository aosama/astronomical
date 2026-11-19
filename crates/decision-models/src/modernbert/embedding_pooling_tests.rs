//! Unit tests for how trunk failures are reported on the embeddings lane.

use astronomical_ipc_protocol::EmbeddingsFailureReason;

use super::{bounded_fatal_execution, trunk_failure};
use crate::modernbert::trunk::ModernBertTrunkError;

fn fatal_reason(failure: EmbeddingsFailureReason) -> String {
    match failure {
        EmbeddingsFailureReason::FatalExecution { reason } => reason,
        other_failure => panic!("expected a fatal execution failure, got {other_failure:?}"),
    }
}

#[test]
fn should_name_the_unavailable_tensor_in_the_lanes_established_wording() {
    let reason = fatal_reason(trunk_failure(ModernBertTrunkError::UnavailableTensor {
        tensor_name: "model.final_norm.weight".to_owned(),
        reason: "not in the artifact".to_owned(),
    }));
    assert_eq!(
        reason,
        "embedding tensor model.final_norm.weight is unavailable: not in the artifact"
    );
}

#[test]
fn should_keep_a_reason_of_exactly_the_cap_and_cut_one_character_more() {
    let reason_at_the_cap = "r".repeat(256);
    assert_eq!(
        fatal_reason(bounded_fatal_execution(&reason_at_the_cap)),
        reason_at_the_cap
    );
    let reason_over_the_cap = "r".repeat(257);
    assert_eq!(
        fatal_reason(bounded_fatal_execution(&reason_over_the_cap))
            .chars()
            .count(),
        256
    );
}

#[test]
fn should_count_the_cap_in_characters_so_a_multibyte_reason_is_never_split() {
    let multibyte_reason = "é".repeat(300);
    let bounded_reason = fatal_reason(bounded_fatal_execution(&multibyte_reason));
    assert_eq!(bounded_reason.chars().count(), 256);
}
