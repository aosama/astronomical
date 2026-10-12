//! HTTP surface: OpenAI-compatible REST endpoints, request translation, and streaming.

pub(crate) mod cache_clear_endpoint;
pub(crate) mod chat_diagnostics;
pub(crate) mod config_reload_endpoint;
pub(crate) mod config_reveal_endpoint;
pub(crate) mod maximum_mlx_memory_endpoint;
pub(crate) mod openai_chat_endpoint;
pub(crate) mod openai_chat_stream;
pub(crate) mod openai_chat_translation;
pub(crate) mod openai_embeddings_endpoint;
pub(crate) mod openai_image_generation_endpoint;
pub(crate) mod openai_models_endpoint;
pub(crate) mod openai_responses_assembly;
pub(crate) mod openai_responses_endpoint;
pub(crate) mod openai_responses_stream;
pub(crate) mod openai_responses_translation;
pub(crate) mod status_endpoint;
