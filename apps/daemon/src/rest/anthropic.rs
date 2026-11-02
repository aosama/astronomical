//! The Anthropic-compatible surface (issue #890): `POST /v1/messages` for
//! Messages-protocol agents such as Claude Code, plus the pure translation,
//! assembly, and streaming-encoding logic it uses. Generation runs on the
//! same worker path as the OpenAI-compatible surface.

pub(crate) mod messages_assembly;
pub(crate) mod messages_endpoint;
pub(crate) mod messages_errors;
pub(crate) mod messages_sse_response;
pub(crate) mod messages_stream;
pub(crate) mod messages_translation;

#[cfg(test)]
mod messages_errors_tests;
#[cfg(test)]
mod messages_stream_tests;
#[cfg(test)]
mod messages_translation_tests;
