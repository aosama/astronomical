//! Prefill-to-decode expert-memory restore for one user request.
//!
//! # What the user is waiting for
//!
//! After the model finishes reading the prompt, it starts writing tokens. Token
//! writing is much cheaper in activation memory than prompt reading. The RAM the
//! user already granted can therefore come back to expert weights so generation
//! does not keep streaming those weights from the solid-state drive.
//!
//! # Words used in this file
//!
//! - Complete residency: every mixture-of-experts weight sits in RAM as one
//!   owner. Decode then never reads experts from disk.
//! - Paged mode: expert weights live on disk. A layer either streams for one
//!   operation or keeps a smaller demand-selected page in RAM.
//! - Demand-selected page: the experts this prompt actually routed to, ranked
//!   by how often they appeared. Not "one routed top-K page per layer".
//! - Temporary request-pressure cap: a smaller retained-page ceiling installed
//!   so the remaining prompt can finish. It is not the user's normal RAM grant.
//!
//! # Why this barrier exists
//!
//! Prefill may demote the complete owner and freeze retained pages so the last
//! prompt chunks still fit. If that freeze is left in place, decode sees a tiny
//! leftover budget, rejects every warm page, and streams from disk even though
//! tens of gigabytes are free. This file is the one place that:
//!
//! 1. Releases the temporary cap after the last prefill cleanup barrier.
//! 2. Tries to restore the complete owner when the leftover ceiling admits it.
//! 3. Otherwise fills demand-selected pages from the full leftover decode budget.
//!
//! Either path is best-effort. A failed restore must not fail the user's
//! request; decode can still stream missing routes.

use astronomical_ipc_protocol::RequestId;

use crate::qwen3_5_moe::{
    Qwen3_5ExpertResidencyPromotionOutcome, Qwen3_5ExpertResidencyTransitionReason,
};

use super::Qwen3_5EngineState;

impl Qwen3_5EngineState {
    /// Restores as much expert RAM as the leftover ceiling admits after prefill.
    ///
    /// Call this exactly once, after the last prefill chunk has synchronized and
    /// cleaned allocator storage, and before the first decode forward. The
    /// request flag `decode_warm_expert_layers_attempted` is the one-shot guard.
    pub(super) fn restore_decode_expert_memory_after_prefill(
        &mut self,
        request_id: RequestId,
        active_request: &mut super::engine_request::Qwen3_5EngineRequest,
    ) {
        let Some(model) = self.model.as_mut() else {
            return;
        };
        // Prefill pressure protects the remaining prompt by installing a
        // temporary retained-page ceiling. That cap must die here. Decode uses
        // a smaller activation footprint, so the leftover composed budget is
        // the user's real grant again. Leaving the cap in place was the bug
        // that kept generation at about one gigabyte of experts.
        let resumed_after_prefill_memory_pressure =
            model.resume_expert_retention_after_request_memory_pressure();
        if resumed_after_prefill_memory_pressure {
            tracing::info!(
                request_id = request_id.value(),
                "released prefill request-pressure expert retention ceiling before decode"
            );
        }
        // Prefer the complete owner. Promotion is replacement-aware: current
        // active memory already owns any hot paged pages, so admission
        // subtracts those pages before adding the complete payload. If this
        // succeeds, decode never needs demand-selected pages.
        if model.sparse_experts_are_paged() {
            match model.try_promote_experts_to_resident(
                Qwen3_5ExpertResidencyTransitionReason::DecodeHandoff,
                &mut active_request.performance_attribution,
            ) {
                Ok(Qwen3_5ExpertResidencyPromotionOutcome::Promoted) => {
                    tracing::info!(
                        request_id = request_id.value(),
                        "restored complete expert residency after prefill before decode"
                    );
                    return;
                }
                Ok(_) => {}
                Err(decode_handoff_promotion_error) => {
                    tracing::info!(
                        request_id = request_id.value(),
                        error = %decode_handoff_promotion_error,
                        "skipped decode-handoff complete residency promotion"
                    );
                }
            }
        }
        // Complete residency did not fit. Spend the leftover decode budget on
        // the experts this prompt actually used. `u64::MAX` means "do not add
        // a second smaller caller cap"; the composed plan is already the cap.
        let context_token_count =
            u64::try_from(active_request.input_token_ids.len()).unwrap_or(u64::MAX);
        match model.fill_retained_expert_pages(
            context_token_count,
            u64::MAX,
            &mut active_request.performance_attribution,
        ) {
            Ok(newly_retained_page_count) => {
                if newly_retained_page_count > 0 {
                    tracing::info!(
                        request_id = request_id.value(),
                        newly_retained_page_count,
                        context_token_count,
                        "decode-warm expert pages ready for generation"
                    );
                }
            }
            Err(decode_warm_fill_error) => {
                tracing::info!(
                    request_id = request_id.value(),
                    error = %decode_warm_fill_error,
                    "skipped decode-warm expert fill; decode will stream routes"
                );
            }
        }
    }
}
