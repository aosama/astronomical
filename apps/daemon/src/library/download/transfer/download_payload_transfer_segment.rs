//! Byte-range segmentation and ordered in-file assembly for one payload file.
//!
//! One ranged stream per whole file leaves the tail of a multi-gigabyte artifact stuck at
//! single-connection throughput. Splitting the remaining range into fixed segments lets several
//! streams fill one file at once, and the assembler below appends those segments strictly in
//! ascending offset order so a staged file's synchronized length always equals its received
//! contiguous prefix — the durability contract that pause and restart recovery reconcile from.

use std::{collections::BTreeMap, path::Path, sync::Arc};

use bytes::Bytes;
use tokio::io::AsyncWriteExt;
use tokio::sync::Mutex;

use super::{
    download_payload_transfer::DownloadPayloadTransferError,
    download_staged_file::open_staged_file_for_append,
};

/// Segment width for one in-file ranged request. Equal segments keep every stream's request
/// latency uniform, and 32 MiB bounds the reorder buffer at one segment's worth per in-flight
/// stream while keeping the count of ranged requests negligible next to the transfer itself.
pub const PAYLOAD_TRANSFER_SEGMENT_BYTES: u64 = 32 * 1024 * 1024;

/// One contiguous absolute byte range of a manifest file.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct PayloadSegment {
    start_offset_bytes: u64,
    end_offset_bytes: u64,
}

impl PayloadSegment {
    #[must_use]
    pub(super) const fn start_offset_bytes(&self) -> u64 {
        self.start_offset_bytes
    }

    #[must_use]
    pub(super) const fn end_offset_bytes(&self) -> u64 {
        self.end_offset_bytes
    }
}

/// Decomposes one file's remaining range into ascending absolute segments.
#[must_use]
pub(super) fn plan_payload_segments(
    resume_offset_bytes: u64,
    expected_file_bytes: u64,
) -> Vec<PayloadSegment> {
    let remaining_bytes = expected_file_bytes - resume_offset_bytes;
    let segment_count = remaining_bytes.div_ceil(PAYLOAD_TRANSFER_SEGMENT_BYTES);
    (0..segment_count)
        .map(|segment_index| {
            let start_offset_bytes =
                resume_offset_bytes + segment_index * PAYLOAD_TRANSFER_SEGMENT_BYTES;
            let end_offset_bytes = start_offset_bytes
                .saturating_add(PAYLOAD_TRANSFER_SEGMENT_BYTES)
                .min(expected_file_bytes);
            PayloadSegment {
                start_offset_bytes,
                end_offset_bytes,
            }
        })
        .collect()
}

struct SegmentAssembly {
    staged_file: tokio::fs::File,
    contiguous_end_bytes: u64,
    buffered_chunks_by_offset: BTreeMap<u64, Bytes>,
}

impl SegmentAssembly {
    async fn append_chunk(
        &mut self,
        chunk_bytes: &[u8],
    ) -> Result<(), DownloadPayloadTransferError> {
        self.staged_file
            .write_all(chunk_bytes)
            .await
            .map_err(DownloadPayloadTransferError::WritePayload)?;
        self.contiguous_end_bytes += chunk_bytes.len() as u64;
        Ok(())
    }

    /// Emits buffered chunks in order while the prefix end matches their start offset.
    async fn drain_buffered_chunks(&mut self) -> Result<(), DownloadPayloadTransferError> {
        while let Some(chunk_bytes) = self
            .buffered_chunks_by_offset
            .remove(&self.contiguous_end_bytes)
        {
            self.append_chunk(&chunk_bytes).await?;
        }
        Ok(())
    }
}

/// Appends concurrently fetched segment bytes into one staged file in ascending offset order.
///
/// Out-of-order chunks park in a bounded buffer (at most one segment's worth per in-flight
/// segment) and drain through the staged file once the prefix reaches them, so disk state never
/// advances past a hole. `finalize` must run before any caller reconciles progress from the
/// staged length.
#[derive(Clone)]
pub(super) struct SegmentedFileAssembler {
    assembly: Arc<Mutex<SegmentAssembly>>,
}

impl SegmentedFileAssembler {
    pub(super) async fn open(
        models_directory: &Path,
        staged_file_path: &Path,
        resume_offset_bytes: u64,
    ) -> Result<Self, DownloadPayloadTransferError> {
        let open_models_directory = models_directory.to_path_buf();
        let open_staged_path = staged_file_path.to_path_buf();
        let staged_file = tokio::task::spawn_blocking(move || {
            open_staged_file_for_append(
                &open_models_directory,
                &open_staged_path,
                resume_offset_bytes,
            )
        })
        .await
        .map_err(DownloadPayloadTransferError::Task)??;
        Ok(Self {
            assembly: Arc::new(Mutex::new(SegmentAssembly {
                staged_file: tokio::fs::File::from_std(staged_file),
                contiguous_end_bytes: resume_offset_bytes,
                buffered_chunks_by_offset: BTreeMap::new(),
            })),
        })
    }

    /// Writes one fetched chunk either directly at the prefix end or into the ordered buffer.
    pub(super) async fn write_chunk(
        &self,
        start_offset_bytes: u64,
        chunk_bytes: Bytes,
    ) -> Result<(), DownloadPayloadTransferError> {
        let mut assembly = self.assembly.lock().await;
        if start_offset_bytes == assembly.contiguous_end_bytes {
            assembly.append_chunk(&chunk_bytes).await?;
            assembly.drain_buffered_chunks().await?;
        } else {
            assembly
                .buffered_chunks_by_offset
                .insert(start_offset_bytes, chunk_bytes);
        }
        Ok(())
    }

    #[must_use]
    pub(super) async fn contiguous_end_bytes(&self) -> u64 {
        self.assembly.lock().await.contiguous_end_bytes
    }

    pub(super) async fn finalize(&self) -> Result<(), DownloadPayloadTransferError> {
        let mut assembly = self.assembly.lock().await;
        assembly.drain_buffered_chunks().await?;
        assembly
            .staged_file
            .sync_all()
            .await
            .map_err(DownloadPayloadTransferError::WritePayload)
    }
}
