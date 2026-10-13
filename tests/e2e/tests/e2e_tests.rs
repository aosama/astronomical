//! Whole-product end-to-end journeys: the real `astronomicald` binary over
//! its real HTTP and IPC surfaces, with fixture worker binaries.
//!
//! Vocabulary (issue #1208): these are **E2E** by seam — every journey spans
//! the process boundary. The daemon's in-process HTTP journeys stay beside
//! the daemon as integration tests; what lives here is exactly the set that
//! treats the daemon as a black-box executable.

mod common;
mod daemon_instance_isolation;
mod daemon_process;
mod library_process;
mod qwen3_5_chat_schema_ipc_journey;
mod qwen_image_21_http_journey;
mod worker_failure_diagnostics;
