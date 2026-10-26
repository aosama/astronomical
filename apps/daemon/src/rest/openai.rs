//! The OpenAI-compatible surface: endpoint wiring for chat, responses,
//! embeddings, image generation, and model listing, plus the pure
//! translation, assembly, and streaming-encoding logic those endpoints use.

pub(crate) mod chat_completion;
pub(crate) mod chat_endpoint;
pub(crate) mod chat_stream;
pub(crate) mod chat_translation;
pub(crate) mod embeddings_endpoint;
pub(crate) mod image_generation_endpoint;
pub(crate) mod models_endpoint;
pub(crate) mod request_body_budget;
pub(crate) mod responses_assembly;
pub(crate) mod responses_endpoint;
pub(crate) mod responses_stream;
pub(crate) mod responses_translation;

#[cfg(test)]
mod chat_translation_tests;
#[cfg(test)]
mod responses_assembly_tests;
#[cfg(test)]
mod responses_stream_tests;
#[cfg(test)]
mod responses_translation_tests;
