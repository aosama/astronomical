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
            Self::Sha256(digest) => is_lowercase_hex(digest, SHA256_HEX_CHARACTER_COUNT),
            Self::GitBlobSha1(digest) => {
                is_lowercase_hex(digest, GIT_BLOB_SHA1_HEX_CHARACTER_COUNT)
            }
        }
    }

    #[must_use]
    pub fn hex(&self) -> &str {
        match self {
            Self::Sha256(digest) | Self::GitBlobSha1(digest) => digest,
        }
    }
}

pub(crate) fn is_lowercase_hex(digest: &str, expected_character_count: usize) -> bool {
    digest.len() == expected_character_count
        && digest
            .bytes()
            .all(|character| character.is_ascii_digit() || (b'a'..=b'f').contains(&character))
}
