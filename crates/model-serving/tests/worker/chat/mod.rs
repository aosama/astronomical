//! Engine-backed chat worker journeys driven over the real IPC framing with
//! scripted engines, factories, and processors.

mod expert_memory_mode;
mod generation_lifecycle;
mod live_memory_limit_wired_budget;
mod model_visible_correction;
mod prefill_progress;
mod prompt_cache_stats;
mod ready_and_model_lifecycle;
mod resident_promotion;
mod resident_promotion_failures;
mod resident_promotion_support;
mod resident_streaming_retry;
pub(super) mod scripted_chat_test_doubles;
mod scripted_model_factory_test_doubles;
pub(super) mod support;
mod tracking_chat_engine_test_doubles;
