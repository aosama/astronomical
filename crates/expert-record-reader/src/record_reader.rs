//! The safe owner of the native record reader.

use std::os::fd::{AsRawFd, BorrowedFd};

/// One record to read: a span of `file` starting at `file_offset`, scattered
/// over `destinations` in order. The span's length is the destinations'
/// total length.
pub struct RecordRead<'destinations> {
    pub file: BorrowedFd<'destinations>,
    pub file_offset: u64,
    pub destinations: Vec<&'destinations mut [u8]>,
}

/// Why a batch of reads did not complete. Destinations of a failed batch
/// hold unspecified bytes.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum RecordReadError {
    #[error("the record reader could not start its threads")]
    ThreadsDidNotStart,
    #[error(
        "a record's file offset {file_offset} or its {destination_count} destinations exceed what one read can address"
    )]
    ReadOutOfRange {
        file_offset: u64,
        destination_count: usize,
    },
    #[error("a record ended before its destinations were full: the file does not hold it")]
    RecordTruncated,
    #[error("a record could not be read: {description} (errno {errno})")]
    ReadFailed { errno: i32, description: String },
}

#[repr(C)]
struct NativeRecordRead {
    file_descriptor: i32,
    destination_count: i32,
    file_offset: i64,
    destinations: *const libc::iovec,
}

#[repr(C)]
struct NativeReader {
    _opaque: [u8; 0],
}

unsafe extern "C" {
    fn astro_expert_record_reader_create(helper_thread_count: u32) -> *mut NativeReader;
    fn astro_expert_record_reader_destroy(reader: *mut NativeReader);
    fn astro_expert_record_reader_read(
        reader: *mut NativeReader,
        reads: *const NativeRecordRead,
        read_count: usize,
    ) -> i32;
}

/// Long-lived threads that read batches of records.
pub struct ExpertRecordReader {
    native_reader: *mut NativeReader,
}

// SAFETY: the native reader guards its state with its own locks and runs
// batches from different threads one after another.
unsafe impl Send for ExpertRecordReader {}
// SAFETY: as above; `read` takes `&self` and the native side serializes it.
unsafe impl Sync for ExpertRecordReader {}

impl ExpertRecordReader {
    /// Starts a reader with `helper_thread_count` helper threads. With none,
    /// every batch is read on the calling thread, one record after another.
    pub fn new(helper_thread_count: u32) -> Result<Self, RecordReadError> {
        // SAFETY: the call takes a plain count and returns an owned handle or null.
        let native_reader = unsafe { astro_expert_record_reader_create(helper_thread_count) };
        if native_reader.is_null() {
            return Err(RecordReadError::ThreadsDidNotStart);
        }
        Ok(Self { native_reader })
    }

    /// Reads every record of the batch and returns when all have finished.
    pub fn read(&self, record_reads: &mut [RecordRead<'_>]) -> Result<(), RecordReadError> {
        let mut destination_vectors: Vec<Vec<libc::iovec>> = Vec::with_capacity(record_reads.len());
        for record_read in record_reads.iter_mut() {
            let mut record_destinations = Vec::with_capacity(record_read.destinations.len());
            for destination in record_read.destinations.iter_mut() {
                record_destinations.push(libc::iovec {
                    iov_base: destination.as_mut_ptr().cast(),
                    iov_len: destination.len(),
                });
            }
            destination_vectors.push(record_destinations);
        }
        let mut native_reads = Vec::with_capacity(record_reads.len());
        for (record_read, record_destinations) in record_reads.iter().zip(&destination_vectors) {
            native_reads.push(native_record_read(record_read, record_destinations)?);
        }
        // SAFETY: every destination is an exclusively borrowed slice that
        // outlives this call, `destination_vectors` outlives it too, and the
        // native call returns only after every read has finished.
        let read_status = unsafe {
            astro_expert_record_reader_read(
                self.native_reader,
                native_reads.as_ptr(),
                native_reads.len(),
            )
        };
        read_result(read_status)
    }
}

fn native_record_read(
    record_read: &RecordRead<'_>,
    record_destinations: &[libc::iovec],
) -> Result<NativeRecordRead, RecordReadError> {
    let out_of_range = RecordReadError::ReadOutOfRange {
        file_offset: record_read.file_offset,
        destination_count: record_destinations.len(),
    };
    let Ok(file_offset) = i64::try_from(record_read.file_offset) else {
        return Err(out_of_range);
    };
    let Ok(destination_count) = i32::try_from(record_destinations.len()) else {
        return Err(out_of_range);
    };
    // preadv refuses more destinations than this, with an error that would
    // read as a failing file.
    if destination_count > libc::IOV_MAX {
        return Err(out_of_range);
    }
    Ok(NativeRecordRead {
        file_descriptor: record_read.file.as_raw_fd(),
        destination_count,
        file_offset,
        destinations: record_destinations.as_ptr(),
    })
}

fn read_result(read_status: i32) -> Result<(), RecordReadError> {
    match read_status {
        0 => Ok(()),
        libc::EIO => Err(RecordReadError::RecordTruncated),
        errno => Err(RecordReadError::ReadFailed {
            errno,
            description: std::io::Error::from_raw_os_error(errno).to_string(),
        }),
    }
}

impl Drop for ExpertRecordReader {
    fn drop(&mut self) {
        // SAFETY: the handle came from `create`, is destroyed once, and no
        // batch can be running because `read` borrows `self`.
        unsafe { astro_expert_record_reader_destroy(self.native_reader) };
    }
}

#[cfg(test)]
#[path = "record_reader_tests.rs"]
mod tests;
