//! OpenAI embeddings response serialization for native local vectors.

use serde::Serialize;

use crate::OpenAiTokenUsage;

/// One completed embedding row in request order.
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct OpenAiEmbedding {
    pub object: &'static str,
    pub index: u32,
    pub embedding: OpenAiEmbeddingVector,
}

/// Float components or base64-encoded Float32 little-endian bytes.
#[derive(Clone, Debug, PartialEq)]
pub enum OpenAiEmbeddingVector {
    Float(Vec<f32>),
    Base64(String),
}

impl Serialize for OpenAiEmbeddingVector {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Float(components) => serializer.serialize_some(components),
            Self::Base64(encoded) => serializer.serialize_some(encoded),
        }
    }
}

impl OpenAiEmbedding {
    /// Builds one embedding row with the OpenAI `embedding` object type.
    #[must_use]
    pub fn new(index: u32, embedding: OpenAiEmbeddingVector) -> Self {
        Self {
            object: "embedding",
            index,
            embedding,
        }
    }
}

/// One OpenAI embeddings list response.
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct OpenAiEmbeddingsResponse {
    pub object: &'static str,
    pub data: Vec<OpenAiEmbedding>,
    pub model: String,
    pub usage: OpenAiTokenUsage,
}

impl OpenAiEmbeddingsResponse {
    /// Serializes the complete validated embedding list with token usage.
    #[must_use]
    pub fn new(embeddings: Vec<OpenAiEmbedding>, model: String, usage: OpenAiTokenUsage) -> Self {
        Self {
            object: "list",
            data: embeddings,
            model,
            usage,
        }
    }
}
