//! Transfer of one payload piece (segment), with recovery.
//!
//! A piece asks for exactly its own byte range, so its response ends where
//! the piece does and the connection stays open for the next piece. A piece
//! whose connection drops, ends early or slows to a trickle is asked for
//! again from the first byte it still lacks, instead of failing the whole
//! download or holding its slot at a crawl.
//!
//! Eight streams stay the width of a download: sixteen measured no faster
//! (333 and 127 Mbit/s against about 360 for eight on a 500 Mbit/s line).

use std::{sync::Arc, time::Duration};

use futures_util::StreamExt;
use tokio::time::Instant;

use super::{
    DownloadPayloadTransfer, DownloadPayloadTransferError,
    payload_response::{payload_url, validate_payload_response},
    payload_transfer_segment::{PayloadSegment, SegmentedFileAssembler},
    payload_transfer_window::{ObservedTransferProgress, SegmentOutcome},
    slow_stream_watch::{SLOW_STREAM_WINDOW, StreamRateReference, bytes_per_second},
};
use crate::library::{
    DownloadJob, DownloadJobFile, DownloadJobStoreError, HubPayloadByteStream, HubPayloadRequest,
};

const LIVE_PROGRESS_PUBLICATION_INTERVAL: Duration = Duration::from_millis(250);

/// How often a piece is asked for again after its connection failed. Three
/// retries ride out a dropped connection or a brief outage; a network that is
/// really gone still fails the download within seconds, with its bytes kept.
const MAXIMUM_FAILED_ATTEMPT_RETRIES: u32 = 3;
const FIRST_RETRY_DELAY: Duration = Duration::from_millis(200);

/// How often a piece's stream is replaced for being slow. Bounded so that a
/// network which has slowed down as a whole is not met with endless
/// reconnecting while the shared rate reference catches up with it.
const MAXIMUM_SLOW_STREAM_RESTARTS: u32 = 2;

/// A measured rate is reported for the reference only from a stream that ran
/// this long; shorter ones are dominated by connection setup.
const MINIMUM_MEASURED_STREAM_DURATION: Duration = Duration::from_secs(1);

/// What one piece transfers into, shared by every attempt at it.
pub(super) struct SegmentTransferTarget<'a> {
    pub(super) download_job: &'a DownloadJob,
    pub(super) download_file: &'a DownloadJobFile,
    pub(super) payload_segment: &'a PayloadSegment,
    pub(super) assembler: Arc<SegmentedFileAssembler>,
    pub(super) updated_at_unix_millis: u64,
    pub(super) observed_transfer_progress: &'a ObservedTransferProgress,
}

/// How one attempt at a piece ended short of an error.
enum SegmentAttemptEnd {
    /// Every byte of the piece arrived.
    Finished,
    /// A pause was requested; what arrived is kept.
    Paused,
    /// The stream fell far behind the download's other streams.
    SlowStream,
}

/// Why an attempt failed, and whether asking again can help.
enum SegmentAttemptError {
    /// The connection failed or the response ended early: ask again.
    Retryable(DownloadPayloadTransferError),
    /// The server's answer or the disk is wrong: asking again cannot help.
    Fatal(DownloadPayloadTransferError),
}

/// One attempt's stream bookkeeping: the bytes of the current watch window
/// and of the whole attempt.
struct StreamMeasurement {
    attempt_started_at: Instant,
    attempt_bytes: u64,
    window_started_at: Instant,
    window_bytes: u64,
}

impl StreamMeasurement {
    fn start() -> Self {
        let started_at = Instant::now();
        Self {
            attempt_started_at: started_at,
            attempt_bytes: 0,
            window_started_at: started_at,
            window_bytes: 0,
        }
    }

    fn record(&mut self, received_bytes: u64) {
        self.attempt_bytes += received_bytes;
        self.window_bytes += received_bytes;
    }

    fn remaining_window(&self) -> Duration {
        SLOW_STREAM_WINDOW.saturating_sub(self.window_started_at.elapsed())
    }

    /// When the watch window has run its length: reports its rate to the
    /// download's reference, starts the next window, and returns whether the
    /// stream ran far behind the others. `false` while the window is open.
    fn close_window_if_due_and_judge_slow(&mut self, rate_reference: &StreamRateReference) -> bool {
        if !self.remaining_window().is_zero() {
            return false;
        }
        let window_duration = self.window_started_at.elapsed();
        // Judged before this window's own rate is folded in, so a slow
        // stream is compared with the others and not with itself.
        let is_slow = rate_reference.is_slow(self.window_bytes, window_duration);
        rate_reference.report(bytes_per_second(self.window_bytes, window_duration));
        self.window_started_at = Instant::now();
        self.window_bytes = 0;
        is_slow
    }
}

impl DownloadPayloadTransfer {
    /// Streams one segment through the file's ordered assembler while
    /// publishing live progress, asking again for the bytes still missing
    /// when an attempt fails or its stream is slow. The assembler appends
    /// only at the contiguous prefix, so the staged file's synchronized
    /// length keeps equaling its received bytes for pause and restart
    /// reconciliation no matter which segment finishes first.
    pub(super) async fn transfer_payload_segment(
        &self,
        segment_target: SegmentTransferTarget<'_>,
    ) -> Result<SegmentOutcome, DownloadPayloadTransferError> {
        let mut transferred_bytes = 0_u64;
        let mut failed_attempt_count = 0_u32;
        let mut slow_stream_restart_count = 0_u32;
        loop {
            let may_restart_slow_stream = slow_stream_restart_count < MAXIMUM_SLOW_STREAM_RESTARTS;
            let attempt_end = self
                .attempt_payload_segment(
                    &segment_target,
                    &mut transferred_bytes,
                    may_restart_slow_stream,
                )
                .await;
            match attempt_end {
                Ok(SegmentAttemptEnd::Finished | SegmentAttemptEnd::Paused) => break,
                Ok(SegmentAttemptEnd::SlowStream) => {
                    slow_stream_restart_count += 1;
                    tracing::info!(
                        relative_path = segment_target.download_file.relative_path(),
                        slow_stream_restart_count,
                        "a download stream fell far behind the others and was replaced"
                    );
                }
                Err(SegmentAttemptError::Retryable(attempt_error))
                    if failed_attempt_count < MAXIMUM_FAILED_ATTEMPT_RETRIES
                        && !self.is_pause_requested() =>
                {
                    failed_attempt_count += 1;
                    tracing::warn!(
                        error = %attempt_error,
                        relative_path = segment_target.download_file.relative_path(),
                        failed_attempt_count,
                        "a download stream failed and is asked for again"
                    );
                    tokio::time::sleep(FIRST_RETRY_DELAY * 2_u32.pow(failed_attempt_count - 1))
                        .await;
                }
                Err(
                    SegmentAttemptError::Retryable(attempt_error)
                    | SegmentAttemptError::Fatal(attempt_error),
                ) => return Err(attempt_error),
            }
        }
        self.segment_outcome(&segment_target).await
    }

    /// The segment's result once its bytes are in: the file's contiguous
    /// prefix, and whether that prefix now covers the whole file. Whichever
    /// segment's write advances the prefix past the file end observes it.
    async fn segment_outcome(
        &self,
        segment_target: &SegmentTransferTarget<'_>,
    ) -> Result<SegmentOutcome, DownloadPayloadTransferError> {
        let contiguous_end_bytes = segment_target.assembler.contiguous_end_bytes().await;
        let file_completed = !self.is_pause_requested()
            && contiguous_end_bytes == segment_target.download_file.expected_bytes();
        if file_completed {
            segment_target.assembler.finalize().await?;
        }
        Ok(SegmentOutcome {
            contiguous_end_bytes,
            file_completed,
        })
    }

    /// One request for the bytes the segment still lacks, streamed until the
    /// segment is complete, a pause is requested, the stream fails, or it
    /// proves slow. `transferred_bytes` carries what earlier attempts already
    /// handed to the assembler, and is advanced by what this one adds.
    async fn attempt_payload_segment(
        &self,
        segment_target: &SegmentTransferTarget<'_>,
        transferred_bytes: &mut u64,
        may_restart_slow_stream: bool,
    ) -> Result<SegmentAttemptEnd, SegmentAttemptError> {
        let mut payload_stream = self
            .open_segment_stream(segment_target, *transferred_bytes)
            .await?;
        let expected_segment_bytes = segment_target.payload_segment.end_offset_bytes()
            - segment_target.payload_segment.start_offset_bytes();
        let rate_reference = segment_target
            .observed_transfer_progress
            .stream_rate_reference();
        let mut stream_measurement = StreamMeasurement::start();
        let mut last_progress_publication_at: Option<Instant> = None;
        loop {
            let next_chunk =
                tokio::time::timeout(stream_measurement.remaining_window(), payload_stream.next())
                    .await;
            if stream_measurement.close_window_if_due_and_judge_slow(rate_reference)
                && may_restart_slow_stream
            {
                return Ok(SegmentAttemptEnd::SlowStream);
            }
            let payload_chunk = match next_chunk {
                // The window ended with nothing new to write. A stalled
                // stream must still notice a pause instead of waiting for
                // its next byte.
                Err(_window_elapsed) if self.is_pause_requested() => {
                    return Ok(SegmentAttemptEnd::Paused);
                }
                Err(_window_elapsed) => continue,
                Ok(None) => break,
                Ok(Some(Err(transport_error))) => {
                    return Err(SegmentAttemptError::Retryable(
                        DownloadPayloadTransferError::Transport(transport_error),
                    ));
                }
                Ok(Some(Ok(payload_chunk))) => payload_chunk,
            };
            let consumed_bytes = write_segment_share_of_chunk(
                segment_target,
                payload_chunk,
                *transferred_bytes,
                expected_segment_bytes,
            )
            .await
            .map_err(SegmentAttemptError::Fatal)?;
            *transferred_bytes += consumed_bytes;
            stream_measurement.record(consumed_bytes);
            self.publish_live_progress_when_due(segment_target, &mut last_progress_publication_at)
                .await
                .map_err(SegmentAttemptError::Fatal)?;
            if *transferred_bytes == expected_segment_bytes {
                report_finished_stream_rate(segment_target, &stream_measurement);
                return Ok(SegmentAttemptEnd::Finished);
            }
            if self.is_pause_requested() {
                return Ok(SegmentAttemptEnd::Paused);
            }
        }
        if self.is_pause_requested() {
            return Ok(SegmentAttemptEnd::Paused);
        }
        // The response ended before the segment did: the connection was cut
        // without an error, so the missing bytes are asked for again.
        Err(SegmentAttemptError::Retryable(
            DownloadPayloadTransferError::InvalidPayloadLength,
        ))
    }

    /// Requests the segment's bytes from `already_transferred_bytes` on and
    /// checks the response's framing before any byte is consumed.
    async fn open_segment_stream(
        &self,
        segment_target: &SegmentTransferTarget<'_>,
        already_transferred_bytes: u64,
    ) -> Result<HubPayloadByteStream, SegmentAttemptError> {
        let request_start_bytes =
            segment_target.payload_segment.start_offset_bytes() + already_transferred_bytes;
        let segment_end_bytes = segment_target.payload_segment.end_offset_bytes();
        let payload_url = payload_url(
            segment_target.download_job.huggingface_id(),
            segment_target.download_job.revision(),
            segment_target.download_file.relative_path(),
        )
        .map_err(SegmentAttemptError::Fatal)?;
        let payload_response = self
            .transport()
            .execute_payload(HubPayloadRequest::get_range(
                payload_url,
                request_start_bytes,
                segment_end_bytes,
            ))
            .await
            .map_err(|transport_error| {
                SegmentAttemptError::Retryable(DownloadPayloadTransferError::Transport(
                    transport_error,
                ))
            })?;
        if matches!(payload_response.status(), 401 | 403) {
            return Err(SegmentAttemptError::Fatal(
                DownloadPayloadTransferError::DownloadGated,
            ));
        }
        validate_payload_response(
            payload_response.status(),
            payload_response.content_range(),
            payload_response.content_length(),
            (request_start_bytes, segment_end_bytes),
            segment_target.download_file.expected_bytes(),
        )
        .map_err(SegmentAttemptError::Fatal)?;
        Ok(payload_response.into_byte_stream())
    }

    /// Publishes the file's contiguous progress at most once per interval, so
    /// the user interface moves without a metadata write per chunk.
    async fn publish_live_progress_when_due(
        &self,
        segment_target: &SegmentTransferTarget<'_>,
        last_progress_publication_at: &mut Option<Instant>,
    ) -> Result<(), DownloadPayloadTransferError> {
        let is_due = last_progress_publication_at.is_none_or(|last_publication_at| {
            last_publication_at.elapsed() >= LIVE_PROGRESS_PUBLICATION_INTERVAL
        });
        if !is_due {
            return Ok(());
        }
        segment_target
            .observed_transfer_progress
            .record_received_bytes(
                segment_target.download_file.relative_path(),
                segment_target.assembler.received_bytes().await,
                segment_target.updated_at_unix_millis,
            )
            .map_err(DownloadJobStoreError::from)?;
        *last_progress_publication_at = Some(Instant::now());
        Ok(())
    }
}

/// Hands the assembler this segment's share of one chunk and returns how
/// many bytes that was. A server that ignores the upper bound sends the rest
/// of the file, so a chunk may span the segment boundary; only the segment's
/// share is consumed and the rest is dropped.
async fn write_segment_share_of_chunk(
    segment_target: &SegmentTransferTarget<'_>,
    payload_chunk: bytes::Bytes,
    transferred_bytes: u64,
    expected_segment_bytes: u64,
) -> Result<u64, DownloadPayloadTransferError> {
    let remaining_segment_bytes = expected_segment_bytes - transferred_bytes;
    let consumed_chunk = if payload_chunk.len() as u64 > remaining_segment_bytes {
        payload_chunk.slice(0..remaining_segment_bytes as usize)
    } else {
        payload_chunk
    };
    let consumed_bytes = consumed_chunk.len() as u64;
    segment_target
        .assembler
        .write_chunk(
            segment_target.payload_segment.start_offset_bytes() + transferred_bytes,
            consumed_chunk,
        )
        .await?;
    Ok(consumed_bytes)
}

/// Folds a finished stream's rate into the download's reference. A healthy
/// piece finishes inside one watch window, so without this the reference
/// would only ever hear from streams slow enough to span a window.
fn report_finished_stream_rate(
    segment_target: &SegmentTransferTarget<'_>,
    stream_measurement: &StreamMeasurement,
) {
    let attempt_duration = stream_measurement.attempt_started_at.elapsed();
    if attempt_duration >= MINIMUM_MEASURED_STREAM_DURATION {
        segment_target
            .observed_transfer_progress
            .stream_rate_reference()
            .report(bytes_per_second(
                stream_measurement.attempt_bytes,
                attempt_duration,
            ));
    }
}
