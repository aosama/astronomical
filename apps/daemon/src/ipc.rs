//! Framed unix-socket IPC service and its per-request handlers.

pub(crate) mod chat;
pub mod chat_schema;
pub(crate) mod decision_training;
pub(crate) mod decisions;
pub(crate) mod embeddings;
pub(crate) mod images;
pub(crate) mod model_unload;
pub(crate) mod models;
pub(crate) mod service;
pub(crate) mod speech;
