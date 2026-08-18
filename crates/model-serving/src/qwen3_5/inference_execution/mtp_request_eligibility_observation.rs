//! Request-level MTP fallback logging and low-overhead numerical attribution.

use astronomical_ipc_protocol::RequestId;

use crate::qwen3_5::multi_token_prediction::Qwen3_5MtpRequestEligibility;
use crate::{PerformanceAttribution, PerformanceCounter};

pub(super) fn record_mtp_request_eligibility(
    request_id: &RequestId,
    mtp_enabled: bool,
    eligibility: Qwen3_5MtpRequestEligibility,
    performance_attribution: &mut PerformanceAttribution,
) {
    tracing::info!(
        request_id = ?request_id,
        mtp_request_eligibility = eligibility.identifier(),
        "evaluated MTP request eligibility"
    );
    if mtp_enabled && !eligibility.is_eligible() {
        performance_attribution
            .record_counter(PerformanceCounter::MtpRequestTargetOnlyFallbackCount, 1);
    }
}
