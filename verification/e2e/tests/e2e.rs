//! Whole-product end-to-end journeys: the real `astronomicald` binary over
//! its real HTTP and IPC surfaces, with fixture worker binaries.
//!
//! Vocabulary (issue #1208): these are **E2E** by seam — every journey spans
//! the process boundary. The daemon's in-process HTTP journeys stay beside
//! the daemon as integration tests; what lives here is exactly the set that
//! treats the daemon as a black-box executable.

mod common;

#[path = "daemon_lifecycle/daemon_instance_isolation.rs"]
mod daemon_instance_isolation;
#[path = "daemon_lifecycle/daemon_process.rs"]
mod daemon_process;
#[path = "model_journeys/decode_full_fit_status_journey.rs"]
mod decode_full_fit_status_journey;
#[path = "library/library_process.rs"]
mod library_process;
#[path = "model_journeys/qwen3_5_chat_schema_ipc_journey.rs"]
mod qwen3_5_chat_schema_ipc_journey;
#[path = "model_journeys/qwen_image_21_http_journey.rs"]
mod qwen_image_21_http_journey;
#[path = "daemon_lifecycle/worker_failure_diagnostics.rs"]
mod worker_failure_diagnostics;
