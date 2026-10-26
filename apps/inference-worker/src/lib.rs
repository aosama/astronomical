//! The inference worker process: launched by the daemon over its IPC pair, it
//! loads a supported model family on the daemon's explicit command and serves
//! decode requests until the daemon closes the transport.
//!
//! Concern map:
//! - `worker_startup` — process bootstrap: diagnostics, the machine MLX
//!   memory ceiling, the Metal wired-residency claim, and the startup/runtime
//!   failure types.
//! - `model_family_factory` — family dispatch into concrete engines
//!   (`#[doc(hidden)]`: consumed by this crate's own integration journeys,
//!   which drive real model directories).
//! - `worker_process_runtime` — Tokio runtime construction with a bounded
//!   shutdown so a finished or failed worker always exits.

#![forbid(unsafe_code)]

#[doc(hidden)]
pub mod model_family_factory;
pub mod worker_process_runtime;
pub mod worker_startup;
