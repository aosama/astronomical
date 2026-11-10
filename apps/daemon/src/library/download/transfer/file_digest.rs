//! Provider-supplied integrity evidence retained across manifest discovery and durable jobs.

use serde::{Deserialize, Serialize};

const GIT_BLOB_SHA1_HEX_CHARACTER_COUNT: usize = 40;
const SHA256_HEX_CHARACTER_COUNT: usize = 64;

/// Digest algorithm dictated by the Hugging Face storage representation of one file.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "algorithm", content = "hex", rename_all = "snake_case")]
pub enum DownloadFileDigest {
    Sha256(String),
    GitBlobSha1(String),
}

impl DownloadFileDigest {
    #[must_use]
    pub fn is_valid(&self) -> bool {
        match self {
            Self::Sha256(digest) => astronomical_lowercase_hex::is_lowercase_hex_of_length(
                digest,
                SHA256_HEX_CHARACTER_COUNT,
            ),
            Self::GitBlobSha1(digest) => astronomical_lowercase_hex::is_lowercase_hex_of_length(
                digest,
                GIT_BLOB_SHA1_HEX_CHARACTER_COUNT,
            ),
        }
    }

    #[must_use]
    pub fn hex(&self) -> &str {
        match self {
            Self::Sha256(digest) | Self::GitBlobSha1(digest) => digest,
        }
    }
}
