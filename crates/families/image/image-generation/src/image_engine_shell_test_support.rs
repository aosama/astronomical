//! What every image family's scripted lifecycle tests share: the event log a
//! fake component owner writes to, how a finalize is recorded, and the
//! memory a finalized request must report. FLUX.2 Klein and Qwen-Image-2.1
//! each kept their own copy of these before.

use std::sync::{Arc, Mutex};

use memory_policy::telemetry::{MlxActiveMemoryBreakdown, MlxMemoryTelemetry};
use model_primitives::performance_attribution::PerformanceAttributionOutcome;

/// The events a fake has recorded so far, in order.
pub fn cloned_events(lifecycle_events: &Arc<Mutex<Vec<String>>>) -> Vec<String> {
    lifecycle_events
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone()
}

/// Records `event`, then fails it when it is the one the test injects a
/// failure at.
pub fn record_lifecycle_event(
    lifecycle_events: &Arc<Mutex<Vec<String>>>,
    failing_event: Option<&str>,
    event: &str,
) -> Result<(), String> {
    lifecycle_events
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .push(event.to_owned());
    if failing_event == Some(event) {
        Err(format!("injected failure at {event}"))
    } else {
        Ok(())
    }
}

/// The event a finalize records: its outcome, then the failure it reports or
/// the encoded byte count.
pub fn finalize_event(
    outcome: PerformanceAttributionOutcome,
    encoded_bytes: Option<u64>,
    failure_description: Option<&str>,
) -> String {
    let outcome_name = match outcome {
        PerformanceAttributionOutcome::Success => "success",
        PerformanceAttributionOutcome::Rejected => "rejected",
        PerformanceAttributionOutcome::Cancelled => "cancelled",
        PerformanceAttributionOutcome::Failed => "failed",
    };
    let outcome_detail = match failure_description {
        Some(failure_description) => failure_description.to_owned(),
        None => encoded_bytes.unwrap_or(0).to_string(),
    };
    format!("finalize:{outcome_name}:{outcome_detail}")
}

/// The memory a fake reports after cleanup: no mlx_buffer_cache bytes left.
pub fn scripted_post_cleanup_memory() -> MlxMemoryTelemetry {
    MlxMemoryTelemetry::new(
        96_000_000,
        0,
        512_000_000,
        MlxActiveMemoryBreakdown::default(),
    )
}

/// A finalized request must report memory with the mlx_buffer_cache emptied.
pub fn assert_final_cleanup_memory(memory_telemetry: Option<MlxMemoryTelemetry>) {
    let memory_telemetry =
        memory_telemetry.expect("finalized image requests should expose MLX memory");
    assert_eq!(memory_telemetry.mlx_buffer_cache_bytes, 0);
    assert!(memory_telemetry.active_memory_bytes <= memory_telemetry.peak_memory_bytes);
    assert_eq!(
        memory_telemetry.active_memory_breakdown,
        MlxActiveMemoryBreakdown::default()
    );
}
