//! Writes every tensor that is not a routed expert into one standard
//! safetensors file, `resident.safetensors`.
//!
//! These are the weights every token needs (attention, norms, embeddings,
//! the language-model head, shared experts, a vision tower), so they form one
//! always-resident class and stay in the format every loader already reads.
//! Tensors are written in name order; bytes are copied unchanged. The bundle
//! is hashed while it is written and again from disk afterwards, and the two
//! digests must agree.

use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::Path;

use sha2::{Digest, Sha256};

use crate::atlas_manifest::{AtlasDeclaredFile, RESIDENT_BUNDLE_FILE_NAME};
use crate::layer_record_writer::SourceShardFiles;
use crate::source_checkpoint::{SourceCheckpoint, SourceTensor};
use crate::uncached_file_io::AlignedSpan;

/// Copies at most this many bytes of one tensor at a time, so a large
/// embedding table never needs one allocation of its own size.
const COPY_PIECE_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Debug, thiserror::Error)]
pub enum ResidentBundleWriteError {
    #[error("{path} could not be accessed: {source}")]
    Io {
        path: String,
        source: std::io::Error,
    },
    #[error("{path} reads back with a different digest than was written")]
    ReadBackMismatch { path: String },
    #[error("the resident bundle header could not be encoded: {0}")]
    Header(#[from] serde_json::Error),
}

/// Writes the bundle into `atlas_directory`, calling `on_bytes_copied` after
/// each copied piece, and returns the file it declares.
pub fn write_resident_bundle(
    source_checkpoint: &SourceCheckpoint,
    atlas_directory: &Path,
    source_shard_files: &mut SourceShardFiles,
    on_bytes_copied: &mut dyn FnMut(u64),
) -> Result<AtlasDeclaredFile, ResidentBundleWriteError> {
    let bundle_path = atlas_directory.join(RESIDENT_BUNDLE_FILE_NAME);
    let path = bundle_path.display().to_string();
    let io_error = |source| ResidentBundleWriteError::Io {
        path: path.clone(),
        source,
    };
    let mut ordered_tensors: Vec<&SourceTensor> =
        source_checkpoint.resident_tensors.iter().collect();
    ordered_tensors.sort_by(|left, right| left.tensor_name.cmp(&right.tensor_name));
    let header_bytes = encode_header(
        &ordered_tensors,
        source_checkpoint.safetensors_metadata.as_ref(),
    )?;
    let mut bundle_writer = HashingWriter::new(File::create_new(&bundle_path).map_err(io_error)?);
    bundle_writer
        .write_all(&(header_bytes.len() as u64).to_le_bytes())
        .map_err(io_error)?;
    bundle_writer.write_all(&header_bytes).map_err(io_error)?;
    for source_tensor in ordered_tensors {
        copy_tensor_payload(
            source_checkpoint,
            source_tensor,
            source_shard_files,
            &mut bundle_writer,
            on_bytes_copied,
        )?;
    }
    let (bundle_byte_count, written_digest) = bundle_writer.finish().map_err(io_error)?;
    if hex_sha256_of_file(&bundle_path).map_err(io_error)? != written_digest {
        return Err(ResidentBundleWriteError::ReadBackMismatch { path });
    }
    Ok(AtlasDeclaredFile {
        file_name: RESIDENT_BUNDLE_FILE_NAME.to_owned(),
        expected_file_byte_count: bundle_byte_count,
        sha256: written_digest,
    })
}

/// The JSON header with packed, name-ordered data offsets.
fn encode_header(
    ordered_tensors: &[&SourceTensor],
    safetensors_metadata: Option<&serde_json::Value>,
) -> Result<Vec<u8>, ResidentBundleWriteError> {
    let mut header_document = serde_json::Map::new();
    if let Some(safetensors_metadata) = safetensors_metadata {
        header_document.insert("__metadata__".to_owned(), safetensors_metadata.clone());
    }
    let mut payload_offset = 0_u64;
    for source_tensor in ordered_tensors {
        let payload_end = payload_offset + source_tensor.payload_byte_count;
        header_document.insert(
            source_tensor.tensor_name.clone(),
            serde_json::json!({
                "dtype": source_tensor.dtype,
                "shape": source_tensor.shape,
                "data_offsets": [payload_offset, payload_end],
            }),
        );
        payload_offset = payload_end;
    }
    Ok(serde_json::to_vec(&serde_json::Value::Object(
        header_document,
    ))?)
}

fn copy_tensor_payload(
    source_checkpoint: &SourceCheckpoint,
    source_tensor: &SourceTensor,
    source_shard_files: &mut SourceShardFiles,
    bundle_writer: &mut HashingWriter,
    on_bytes_copied: &mut dyn FnMut(u64),
) -> Result<(), ResidentBundleWriteError> {
    let shard_path = source_checkpoint
        .model_directory
        .join(&source_tensor.shard_file_name);
    let io_error = |source| ResidentBundleWriteError::Io {
        path: shard_path.display().to_string(),
        source,
    };
    let shard_file = source_shard_files
        .file(
            &source_checkpoint.model_directory,
            &source_tensor.shard_file_name,
        )
        .map_err(io_error)?;
    let mut copied_byte_count = 0;
    while copied_byte_count < source_tensor.payload_byte_count {
        let piece_byte_count =
            COPY_PIECE_BYTES.min(source_tensor.payload_byte_count - copied_byte_count);
        let tensor_piece = AlignedSpan::read(
            shard_file,
            source_tensor.payload_file_offset + copied_byte_count,
            piece_byte_count,
        )
        .map_err(io_error)?;
        bundle_writer
            .write_all(tensor_piece.bytes())
            .map_err(io_error)?;
        copied_byte_count += piece_byte_count;
        on_bytes_copied(piece_byte_count);
    }
    Ok(())
}

/// A buffered file writer that hashes everything it writes.
struct HashingWriter {
    file_writer: BufWriter<File>,
    digest: Sha256,
    written_byte_count: u64,
}

impl HashingWriter {
    fn new(bundle_file: File) -> Self {
        Self {
            file_writer: BufWriter::with_capacity(8 << 20, bundle_file),
            digest: Sha256::new(),
            written_byte_count: 0,
        }
    }

    fn write_all(&mut self, written_bytes: &[u8]) -> std::io::Result<()> {
        self.file_writer.write_all(written_bytes)?;
        self.digest.update(written_bytes);
        self.written_byte_count += written_bytes.len() as u64;
        Ok(())
    }

    fn finish(self) -> std::io::Result<(u64, String)> {
        let bundle_file = self
            .file_writer
            .into_inner()
            .map_err(std::io::IntoInnerError::into_error)?;
        bundle_file.sync_all()?;
        let digest_hex = self
            .digest
            .finalize()
            .iter()
            .map(|digest_byte| format!("{digest_byte:02x}"))
            .collect();
        Ok((self.written_byte_count, digest_hex))
    }
}

fn hex_sha256_of_file(path: &Path) -> std::io::Result<String> {
    let mut written_file = File::open(path)?;
    let mut digest = Sha256::new();
    let mut read_piece = vec![0u8; 8 << 20];
    loop {
        let read_byte_count = written_file.read(&mut read_piece)?;
        if read_byte_count == 0 {
            break;
        }
        digest.update(&read_piece[..read_byte_count]);
    }
    Ok(digest
        .finalize()
        .iter()
        .map(|digest_byte| format!("{digest_byte:02x}"))
        .collect())
}
