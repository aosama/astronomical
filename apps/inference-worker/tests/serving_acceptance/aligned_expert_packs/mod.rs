//! User-journey acceptance for the converted per-expert streaming model.
//!
//! The converted revision is a separate discoverable Qwen3.5-MoE identity
//! produced by `astronomical-experimental-aligned-expert-pack-preparer
//! --streaming-model`. It is single-copy: every weight byte lives exactly
//! once — non-expert tensors in `resident.safetensors`, expert bytes only in
//! the per-expert `.apack` files — and discovery, validation, weight binding,
//! and expert paging all operate through the revision's own manifest.

mod advertise;
mod chat;
mod support;
mod tools;
