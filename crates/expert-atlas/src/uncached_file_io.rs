//! Reads and writes that bypass the macOS file cache.
//!
//! Measured on Apple Silicon: `F_NOCACHE` gives a direct (uncached) transfer
//! only when the file offset, the length and the buffer are all 16 KiB
//! aligned; otherwise it silently goes through the cache. A conversion reads
//! and writes tens of gigabytes once, so keeping it out of the cache leaves
//! the cache to the models the machine is serving.

use std::fs::{File, OpenOptions};
use std::os::fd::AsRawFd;
use std::os::unix::fs::FileExt;
use std::path::Path;

use crate::atlas_manifest::ATLAS_RECORD_ALIGNMENT_BYTES;

pub(crate) fn open_uncached(path: &Path) -> std::io::Result<File> {
    let file = File::open(path)?;
    set_no_cache(&file)?;
    Ok(file)
}

pub(crate) fn create_uncached(path: &Path) -> std::io::Result<File> {
    let file = OpenOptions::new()
        .create_new(true)
        .read(true)
        .write(true)
        .open(path)?;
    set_no_cache(&file)?;
    Ok(file)
}

fn set_no_cache(file: &File) -> std::io::Result<()> {
    // SAFETY: fcntl on a live descriptor with an integer argument.
    let status = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1) };
    if status == -1 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(())
    }
}

/// A zeroed buffer whose start and length are 16 KiB aligned.
pub(crate) struct PageAlignedBuffer {
    pointer: *mut u8,
    layout: std::alloc::Layout,
}

impl PageAlignedBuffer {
    pub(crate) fn new(byte_count: u64) -> Self {
        let rounded_byte_count =
            byte_count.max(1).div_ceil(ATLAS_RECORD_ALIGNMENT_BYTES) * ATLAS_RECORD_ALIGNMENT_BYTES;
        let layout = std::alloc::Layout::from_size_align(
            rounded_byte_count as usize,
            ATLAS_RECORD_ALIGNMENT_BYTES as usize,
        )
        .expect("a page-aligned layout of a file-sized buffer is valid");
        // SAFETY: the size is at least one page and the alignment a power of two.
        let pointer = unsafe { std::alloc::alloc_zeroed(layout) };
        if pointer.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        Self { pointer, layout }
    }

    pub(crate) fn bytes(&self) -> &[u8] {
        // SAFETY: the allocation spans layout.size() initialized bytes.
        unsafe { std::slice::from_raw_parts(self.pointer, self.layout.size()) }
    }

    pub(crate) fn bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: the allocation spans layout.size() initialized bytes.
        unsafe { std::slice::from_raw_parts_mut(self.pointer, self.layout.size()) }
    }
}

impl Drop for PageAlignedBuffer {
    fn drop(&mut self) {
        // SAFETY: allocated in `new` with this exact layout.
        unsafe { std::alloc::dealloc(self.pointer, self.layout) };
    }
}

// SAFETY: the buffer exclusively owns its allocation.
unsafe impl Send for PageAlignedBuffer {}

/// The bytes `[offset, offset + length)` of `file`, read through their
/// enclosing aligned span so the read is direct.
pub(crate) struct AlignedSpan {
    aligned_buffer: PageAlignedBuffer,
    skipped_byte_count: usize,
    span_byte_count: usize,
}

impl AlignedSpan {
    pub(crate) fn read(file: &File, offset: u64, length: u64) -> std::io::Result<Self> {
        let file_length = file.metadata()?.len();
        if offset + length > file_length {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!("span {offset}+{length} runs past the end of a {file_length}-byte file"),
            ));
        }
        let alignment = ATLAS_RECORD_ALIGNMENT_BYTES;
        let aligned_start = offset / alignment * alignment;
        let aligned_length = (offset + length - aligned_start).div_ceil(alignment) * alignment;
        let readable_length = aligned_length.min(file_length - aligned_start) as usize;
        let mut aligned_buffer = PageAlignedBuffer::new(aligned_length);
        file.read_exact_at(
            &mut aligned_buffer.bytes_mut()[..readable_length],
            aligned_start,
        )?;
        Ok(Self {
            aligned_buffer,
            skipped_byte_count: (offset - aligned_start) as usize,
            span_byte_count: length as usize,
        })
    }

    pub(crate) fn bytes(&self) -> &[u8] {
        &self.aligned_buffer.bytes()
            [self.skipped_byte_count..self.skipped_byte_count + self.span_byte_count]
    }
}

/// Bytes available to an unprivileged writer on the volume holding `path`.
pub(crate) fn available_bytes_on_volume(path: &Path) -> std::io::Result<u64> {
    let path_bytes = std::ffi::CString::new(path.as_os_str().as_encoded_bytes()).map_err(|_| {
        std::io::Error::new(std::io::ErrorKind::InvalidInput, "path has a null byte")
    })?;
    // SAFETY: statvfs fills a zeroed struct for a valid, null-terminated path.
    let mut volume_statistics: libc::statvfs = unsafe { std::mem::zeroed() };
    // SAFETY: as above; the struct outlives the call.
    let status = unsafe { libc::statvfs(path_bytes.as_ptr(), &mut volume_statistics) };
    if status != 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(u64::from(volume_statistics.f_bavail) * volume_statistics.f_frsize)
}
