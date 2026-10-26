//! Framed unix-socket IPC service and its per-request handlers.

pub(crate) mod chat;
pub mod chat_schema;
pub(crate) mod embeddings;
pub(crate) mod models;
pub(crate) mod service;
