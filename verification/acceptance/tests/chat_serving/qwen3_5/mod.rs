//! Serving-acceptance journeys for the Qwen3.5-family sparse-MoE 35B-A3B artifacts.

mod acceptance_worker_config;
mod automatic_engine_selection_rest;
mod live_ceiling_lowering_rest;
mod one_token_chat_request;
mod sparse_artifact_residency;
mod speculative_decoding_rest;
