use super::{invalid_request_error, resident_capacity_error};
use crate::{InferenceEngineError, StreamingEngineRequirement};

#[test]
fn should_send_a_request_that_does_not_fit_to_the_streaming_engine_with_its_cause() {
    let capacity_error = resident_capacity_error(
        StreamingEngineRequirement::StableShortfall { shortfall_bytes: 7 },
        "the weights and the stored state exceed the limit",
    );
    assert!(matches!(
        capacity_error,
        InferenceEngineError::ResidentForkRequired {
            cause: StreamingEngineRequirement::StableShortfall { shortfall_bytes: 7 },
            ref reason,
        } if reason == "the weights and the stored state exceed the limit"
    ));
}

#[test]
fn should_never_report_a_request_that_does_not_fit_as_invalid() {
    for cause in [
        StreamingEngineRequirement::RequestContextShortfall { shortfall_bytes: 0 },
        StreamingEngineRequirement::TransientPeakShortfallAtPrefillChunk {
            shortfall_bytes: usize::MAX,
            prefill_chunk_token_count: 1,
        },
        StreamingEngineRequirement::ForwardGraphicsProcessorMemoryExhausted,
    ] {
        assert!(!matches!(
            resident_capacity_error(cause, "does not fit"),
            InferenceEngineError::InvalidRequest { .. }
        ));
    }
}

#[test]
fn should_keep_a_malformed_request_invalid() {
    assert!(matches!(
        invalid_request_error("the request names no tokens"),
        InferenceEngineError::InvalidRequest { ref reason } if reason == "the request names no tokens"
    ));
}
