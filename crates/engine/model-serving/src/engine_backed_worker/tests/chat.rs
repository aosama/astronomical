//! Engine-backed chat worker journeys driven over the real IPC framing with
//! scripted engines, factories, and processors.

mod astro_ssd_prompt_cache_stats;
mod chat_request_exchange;
mod count_chat_tokens;
mod expert_memory_mode;
mod generation_lifecycle;
pub(super) mod ipc_worker_transport;
mod live_memory_limit_wired_budget;
mod model_visible_correction;
mod neural_engine_preparation;
mod prefill_progress;
mod ready_and_model_lifecycle;
pub(super) mod scripted_chat_test_doubles;
mod scripted_model_factory_test_doubles;
pub(super) mod support;
mod tracking_chat_engine_test_doubles;
