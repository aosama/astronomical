//! The non-weight files a conversion copies verbatim from the source so the
//! output loads with its tokenizer and chat template. The patterns are
//! OptiQ's (`optiq/core/shardwise.py`).

/// The files copied verbatim from source to output beside the shards.
pub const DEFAULT_COPY_PATTERNS: &[&str] = &[
    "tokenizer*",
    "*.jinja",
    "generation_config.json",
    "chat_template*",
    "preprocessor_config.json",
    "special_tokens_map.json",
];
