//! The loopback REST surface: OpenAI-compatible endpoints, the daemon
//! control endpoints, and the pure request/response translation between the
//! REST DTOs and the worker's commands and events.
//!
//! `openai/` holds the OpenAI-compatible family and `anthropic/` the
//! Anthropic Messages family: endpoint wiring, request translation, response
//! assembly, and SSE encoding. The flat modules are
//! the daemon control surface (status, configuration reload, cache, memory
//! ceiling) and the request-diagnostics helper shared by both families.

pub(crate) mod anthropic;
pub(crate) mod openai;

pub(crate) mod cache_clear;
pub(crate) mod chat_diagnostics;
pub(crate) mod chronological_instruction;
pub(crate) mod config_document;
pub(crate) mod config_reload;
pub(crate) mod config_reload_memory;
pub(crate) mod config_reveal;
pub(crate) mod expert_placement;
pub(crate) mod experts;
pub(crate) mod finder_reveal;
pub(crate) mod loopback_browser_guard;
pub(crate) mod maximum_mlx_memory;
pub(crate) mod request_body_budget;
pub(crate) mod request_rejections;
pub(crate) mod status;
pub(crate) mod system_one_endpoint;

#[cfg(test)]
mod chat_diagnostics_tests;
#[cfg(test)]
mod config_reload_tests;
#[cfg(test)]
mod expert_placement_tests;
#[cfg(test)]
mod experts_tests;
