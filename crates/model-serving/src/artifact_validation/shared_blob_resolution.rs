//! Verified resolution of Hugging Face shared cache blobs.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use super::ArtifactValidationError;

/// Directory, directly under a Hugging Face hub root, that holds shared blobs.
const SHARED_BLOB_STORE_DIRECTORY_NAME: &str = "blobs";

/// Directory, inside one cache entry, that holds the per-revision tree metadata.
const TREES_DIRECTORY_NAME: &str = "trees";

/// Snapshot tree key holding the per-file record map.
const TREE_FILES_KEY: &str = "files";

/// Snapshot tree key holding a file's plain byte size.
const TREE_SIZE_KEY: &str = "size";

/// Snapshot tree key holding a classic LFS blob's SHA-256 digest.
const TREE_LFS_SHA256_KEY: &str = "lfs_sha256";

/// Snapshot tree key holding a Xet-backed blob's content hash.
const TREE_XET_HASH_KEY: &str = "xet_hash";

/// A snapshot tree record for a small cache never approaches this bound; the
/// limit exists so a corrupt or hostile tree file cannot be read into memory
/// without limit.
const MAXIMUM_TREE_METADATA_BYTES: u64 = 8 * 1024 * 1024;

const SHA256_DIGEST_HEX_CHARACTERS: usize = 64;

/// Number of leading digest characters used as the shared store's subdirectory.
const CONTENT_ADDRESS_DIRECTORY_HEX_CHARACTERS: usize = 2;

/// Resolves `canonical_resolved_target_path` as a verified shared Hugging Face
/// cache blob when the snapshot symlink for `required_file_name` legitimately
/// reaches the hub-level blob store.
///
/// Returns `Ok(None)` when the target is not inside a hub-level shared blob
/// store, which leaves the caller's own confinement decision in charge.
///
/// Verification is provenance-based rather than a byte re-hash. The store names
/// every object after the digest recorded in the snapshot tree, so accepting a
/// target requires all of: the store is a real (non-symlink) directory, the
/// target sits inside it under the recorded content address, the target is a
/// regular file, and its length matches the recorded size. Re-hashing would add
/// a second full read of multi-gigabyte shards on the model-load critical path,
/// which this codebase deliberately avoids.
pub(crate) fn resolve_verified_shared_hub_blob_path(
    hub_root_directory: &Path,
    snapshot_directory: &Path,
    canonical_resolved_target_path: &Path,
    required_file_name: &str,
) -> Result<Option<PathBuf>, ArtifactValidationError> {
    let shared_blob_store_directory = hub_root_directory.join(SHARED_BLOB_STORE_DIRECTORY_NAME);
    let Ok(shared_blob_store_metadata) = fs::symlink_metadata(&shared_blob_store_directory) else {
        return Ok(None);
    };
    if shared_blob_store_metadata.file_type().is_symlink() || !shared_blob_store_metadata.is_dir() {
        return Ok(None);
    }
    let canonical_shared_blob_store_directory = fs::canonicalize(&shared_blob_store_directory)
        .map_err(|source| unavailable_shared_blob_metadata(required_file_name, source))?;
    if !canonical_resolved_target_path.starts_with(&canonical_shared_blob_store_directory) {
        return Ok(None);
    }

    let snapshot_tree_record = read_snapshot_tree_record(snapshot_directory, required_file_name)?;
    let is_recorded_content_addressed_object = snapshot_tree_record
        .recorded_content_digests
        .iter()
        .any(|recorded_content_digest| {
            content_addressed_blob_path_matches(
                &canonical_shared_blob_store_directory,
                recorded_content_digest,
                canonical_resolved_target_path,
            )
        });
    if !is_recorded_content_addressed_object {
        return Err(
            ArtifactValidationError::HuggingFaceSharedBlobIdentityMismatch {
                file_name: required_file_name.to_owned(),
                recorded_digest_text: snapshot_tree_record.primary_recorded_digest,
            },
        );
    }

    let shared_blob_metadata = fs::symlink_metadata(canonical_resolved_target_path)
        .map_err(|source| unavailable_shared_blob_metadata(required_file_name, source))?;
    if !shared_blob_metadata.file_type().is_file() {
        return Err(ArtifactValidationError::RequiredFileIsNotRegular {
            file_name: required_file_name.to_owned(),
        });
    }
    if shared_blob_metadata.len() != snapshot_tree_record.recorded_size_bytes {
        return Err(ArtifactValidationError::HuggingFaceSharedBlobSizeMismatch {
            file_name: required_file_name.to_owned(),
            recorded_size_bytes: snapshot_tree_record.recorded_size_bytes,
            actual_size_bytes: shared_blob_metadata.len(),
        });
    }

    tracing::debug!(
        file_name = %required_file_name,
        shared_blob_path = %canonical_resolved_target_path.display(),
        recorded_digest_text = %snapshot_tree_record.primary_recorded_digest,
        recorded_size_bytes = snapshot_tree_record.recorded_size_bytes,
        "verified Hugging Face shared cache blob for required file"
    );
    Ok(Some(canonical_resolved_target_path.to_path_buf()))
}

/// One file record from a Hugging Face cache entry's snapshot tree metadata.
struct SnapshotTreeRecord {
    /// Digest the shared store names this object after, preferring the Xet hash
    /// because Xet-backed caches store the object under that name.
    primary_recorded_digest: String,
    /// Every digest the tree records for this file, in preference order.
    recorded_content_digests: Vec<String>,
    /// Exact byte length the tree records for this file.
    recorded_size_bytes: u64,
}

/// Reads the snapshot tree record that anchors trust for one cache file.
///
/// The tree metadata is written by the Hugging Face client next to the blobs and
/// is not part of the snapshot itself, so a snapshot whose tree record is
/// missing or unreadable is rejected rather than trusted on name alone.
fn read_snapshot_tree_record(
    snapshot_directory: &Path,
    required_file_name: &str,
) -> Result<SnapshotTreeRecord, ArtifactValidationError> {
    let tree_metadata_path = snapshot_tree_metadata_path(snapshot_directory, required_file_name)?;
    let tree_metadata_bytes = read_bounded_tree_metadata(&tree_metadata_path, required_file_name)?;
    let tree_metadata: serde_json::Value =
        serde_json::from_slice(&tree_metadata_bytes).map_err(|source| {
            unavailable_shared_blob_metadata(required_file_name, io::Error::other(source))
        })?;
    let file_record = tree_metadata
        .get(TREE_FILES_KEY)
        .and_then(|files| files.get(required_file_name))
        .ok_or_else(|| {
            unavailable_shared_blob_metadata(
                required_file_name,
                io::Error::new(
                    io::ErrorKind::NotFound,
                    "snapshot tree metadata has no record for this file",
                ),
            )
        })?;
    let recorded_size_bytes = file_record
        .get(TREE_SIZE_KEY)
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| {
            unavailable_shared_blob_metadata(
                required_file_name,
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "snapshot tree record has no byte size",
                ),
            )
        })?;
    let recorded_content_digests = [TREE_XET_HASH_KEY, TREE_LFS_SHA256_KEY]
        .into_iter()
        .filter_map(|digest_key| file_record.get(digest_key))
        .filter_map(serde_json::Value::as_str)
        .map(|digest_text| decode_content_digest(digest_text, required_file_name))
        .collect::<Result<Vec<String>, ArtifactValidationError>>()?;
    let Some(primary_recorded_digest) = recorded_content_digests.first().cloned() else {
        return Err(unavailable_shared_blob_metadata(
            required_file_name,
            io::Error::new(
                io::ErrorKind::InvalidData,
                "snapshot tree record has no content digest",
            ),
        ));
    };

    Ok(SnapshotTreeRecord {
        primary_recorded_digest,
        recorded_content_digests,
        recorded_size_bytes,
    })
}

fn snapshot_tree_metadata_path(
    snapshot_directory: &Path,
    required_file_name: &str,
) -> Result<PathBuf, ArtifactValidationError> {
    let revision_name = snapshot_directory
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| {
            unavailable_shared_blob_metadata(
                required_file_name,
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "snapshot directory has no name",
                ),
            )
        })?;
    let model_cache_directory = snapshot_directory
        .parent()
        .and_then(Path::parent)
        .ok_or_else(|| {
            unavailable_shared_blob_metadata(
                required_file_name,
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "snapshot directory has no cache entry parent",
                ),
            )
        })?;
    Ok(model_cache_directory
        .join(TREES_DIRECTORY_NAME)
        .join(format!("{revision_name}.json")))
}

fn read_bounded_tree_metadata(
    tree_metadata_path: &Path,
    required_file_name: &str,
) -> Result<Vec<u8>, ArtifactValidationError> {
    let tree_metadata_len = fs::metadata(tree_metadata_path)
        .map_err(|source| unavailable_shared_blob_metadata(required_file_name, source))?
        .len();
    if tree_metadata_len > MAXIMUM_TREE_METADATA_BYTES {
        return Err(unavailable_shared_blob_metadata(
            required_file_name,
            io::Error::new(
                io::ErrorKind::InvalidData,
                "snapshot tree metadata is larger than the accepted bound",
            ),
        ));
    }
    fs::read(tree_metadata_path)
        .map_err(|source| unavailable_shared_blob_metadata(required_file_name, source))
}

/// Accepts only a lowercase hexadecimal content digest of the length the shared
/// store's directory layout assumes.
fn decode_content_digest(
    digest_text: &str,
    required_file_name: &str,
) -> Result<String, ArtifactValidationError> {
    let is_lowercase_hexadecimal = digest_text.chars().all(|digest_character| {
        digest_character.is_ascii_digit() || ('a'..='f').contains(&digest_character)
    });
    if digest_text.len() != SHA256_DIGEST_HEX_CHARACTERS || !is_lowercase_hexadecimal {
        return Err(unavailable_shared_blob_metadata(
            required_file_name,
            io::Error::new(
                io::ErrorKind::InvalidData,
                "snapshot tree record has a malformed content digest",
            ),
        ));
    }
    Ok(digest_text.to_owned())
}

/// Checks both Hugging Face shared-store filename layouts.
///
/// Classic LFS blobs use the digest remainder after the two-character directory
/// prefix. Xet-backed blobs retain the complete digest as the filename. Both
/// layouts are constrained to the canonical hub-level store and the digest
/// recorded by the snapshot tree.
fn content_addressed_blob_path_matches(
    shared_blob_store_directory: &Path,
    content_digest: &str,
    canonical_resolved_target_path: &Path,
) -> bool {
    let Some(directory_prefix) = content_digest.get(..CONTENT_ADDRESS_DIRECTORY_HEX_CHARACTERS)
    else {
        return false;
    };
    let Some(digest_remainder) = content_digest.get(CONTENT_ADDRESS_DIRECTORY_HEX_CHARACTERS..)
    else {
        return false;
    };
    let prefixed_store_path = shared_blob_store_directory
        .join(directory_prefix)
        .join(digest_remainder);
    let full_digest_store_path = shared_blob_store_directory
        .join(directory_prefix)
        .join(content_digest);
    canonical_resolved_target_path == prefixed_store_path
        || canonical_resolved_target_path == full_digest_store_path
}

fn unavailable_shared_blob_metadata(
    required_file_name: &str,
    source: io::Error,
) -> ArtifactValidationError {
    ArtifactValidationError::HuggingFaceSharedBlobMetadataUnavailable {
        file_name: required_file_name.to_owned(),
        source,
    }
}
