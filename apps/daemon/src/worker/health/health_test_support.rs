//! Set-up shared by the health snapshot tests.

use ipc_protocol::WorkerModelCapabilities;

use super::WorkerHealthSnapshot;

/// A model that serves nothing a capability gates.
pub(super) fn no_capabilities() -> WorkerModelCapabilities {
    WorkerModelCapabilities {
        chat: None,
        image_generation: None,
        embeddings: None,
        decisions: None,
        speech: None,
    }
}

/// A snapshot whose worker is ready with one model loaded.
pub(super) fn ready_snapshot() -> WorkerHealthSnapshot {
    WorkerHealthSnapshot::ready_with_model(String::from("first-model"), no_capabilities())
}
