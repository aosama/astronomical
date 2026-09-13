//! The `qwen4_exp` model family (Qwen 3.8 Flash).
//!
//! This family is under construction: recognition, artifact validation, and
//! execution land in separate steps, and only the pieces that exist are
//! exported. Everything in here is held to the repository's standing rules
//! for the family — see `docs/qwen4-exp-architecture.md` for the evidence,
//! the naming ruling, and the artifact-independence contract.
//!
//! Current contents:
//! - `configuration` — validated text configuration for the family.
//! - `decoder` — per-request state streams: layout and byte geometry.
//! - `ple` — row identity for the hashed n-gram embedding table.
//! - `hyper_connection` — the residual stream mixing algebra.

pub mod configuration;
pub mod decoder;
pub mod hyper_connection;
pub mod ple;
pub mod qsa;
