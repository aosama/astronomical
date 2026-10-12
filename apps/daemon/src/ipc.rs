//! Framed unix-socket IPC service and its per-request handlers.

pub(crate) mod daemon_ipc;
pub(crate) mod daemon_ipc_chat;
pub mod daemon_ipc_chat_schema;
pub(crate) mod daemon_ipc_embeddings;
pub(crate) mod daemon_ipc_models;
