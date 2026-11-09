//! File reads that reach the SSD rather than the macOS file cache, the check
//! that proves a range was not cached, and the scatter read the atlas uses.
//!
//! Measured on Apple Silicon with this probe's own reads: `F_NOCACHE` gives a
//! direct (uncached) read only when the file offset, the length and the
//! destination are all 16 KiB aligned (the virtual-memory page size), and a
//! repeated aligned read stays cold; an unaligned read is silently served
//! through the cache and leaves the range cached. Stock safetensors tensors
//! start at unaligned offsets, so a stock-layout read always warms the cache.

use std::fs::{File, OpenOptions};
use std::os::fd::AsRawFd;
use std::os::unix::fs::FileExt;
use std::path::Path;

/// Alignment of offset, length and destination that a direct read needs.
pub(super) const DIRECT_READ_ALIGNMENT_BYTES: u64 = 16_384;

pub(super) fn open_uncached(path: &Path) -> File {
    let file =
        File::open(path).unwrap_or_else(|error| panic!("{} should open: {error}", path.display()));
    set_no_cache(&file);
    file
}

pub(super) fn create_uncached(path: &Path) -> File {
    let file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .read(true)
        .write(true)
        .open(path)
        .unwrap_or_else(|error| panic!("{} should be creatable: {error}", path.display()));
    set_no_cache(&file);
    file
}

fn set_no_cache(file: &File) {
    // SAFETY: fcntl on a live descriptor with an integer argument.
    let status = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1) };
    assert_ne!(
        status,
        -1,
        "F_NOCACHE should apply: {}",
        std::io::Error::last_os_error()
    );
}

/// A zeroed allocation whose start is aligned for direct reads.
pub(super) struct PageAlignedBuffer {
    pointer: *mut u8,
    layout: std::alloc::Layout,
}

impl PageAlignedBuffer {
    pub(super) fn new(byte_count: u64) -> Self {
        assert!(byte_count > 0, "an aligned buffer needs at least one byte");
        let rounded_byte_count =
            byte_count.div_ceil(DIRECT_READ_ALIGNMENT_BYTES) * DIRECT_READ_ALIGNMENT_BYTES;
        let layout = std::alloc::Layout::from_size_align(
            rounded_byte_count as usize,
            DIRECT_READ_ALIGNMENT_BYTES as usize,
        )
        .expect("a page-aligned layout is valid");
        // SAFETY: the size is non-zero (asserted above) and the alignment is a power of two.
        let pointer = unsafe { std::alloc::alloc_zeroed(layout) };
        assert!(!pointer.is_null(), "the aligned allocation should succeed");
        Self { pointer, layout }
    }

    pub(super) fn bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: the allocation spans layout.size() zero-initialized bytes.
        unsafe { std::slice::from_raw_parts_mut(self.pointer, self.layout.size()) }
    }
}

impl Drop for PageAlignedBuffer {
    fn drop(&mut self) {
        // SAFETY: allocated in `new` with this exact layout.
        unsafe { std::alloc::dealloc(self.pointer, self.layout) };
    }
}

/// Reads `[offset, offset + length)` through its enclosing aligned span, so
/// the read is direct and leaves the source cold for later measurements.
pub(super) fn read_cold_span(file: &File, offset: u64, length: u64) -> PageAlignedSpan {
    assert!(length > 0, "a cold span read needs at least one byte");
    let file_length = file.metadata().expect("source metadata").len();
    assert!(
        offset + length <= file_length,
        "the span lies inside the file"
    );
    let aligned_start = offset / DIRECT_READ_ALIGNMENT_BYTES * DIRECT_READ_ALIGNMENT_BYTES;
    let aligned_length = (offset + length - aligned_start).div_ceil(DIRECT_READ_ALIGNMENT_BYTES)
        * DIRECT_READ_ALIGNMENT_BYTES;
    let readable_length = aligned_length.min(file_length - aligned_start) as usize;
    let mut aligned_buffer = PageAlignedBuffer::new(aligned_length);
    file.read_exact_at(
        &mut aligned_buffer.bytes_mut()[..readable_length],
        aligned_start,
    )
    .expect("the aligned source span should read");
    PageAlignedSpan {
        aligned_buffer,
        skipped_byte_count: (offset - aligned_start) as usize,
        span_byte_count: length as usize,
    }
}

/// The requested bytes inside the aligned buffer they were read into,
/// borrowed rather than copied out.
pub(super) struct PageAlignedSpan {
    aligned_buffer: PageAlignedBuffer,
    skipped_byte_count: usize,
    span_byte_count: usize,
}

impl PageAlignedSpan {
    pub(super) fn bytes(&mut self) -> &[u8] {
        let span_start = self.skipped_byte_count;
        &self.aligned_buffer.bytes_mut()[span_start..span_start + self.span_byte_count]
    }
}

/// Whether any virtual-memory page under the range is in the file cache.
pub(super) fn any_vm_page_resident(file: &File, offset: u64, length: u64) -> bool {
    assert!(length > 0, "a residency check needs at least one byte");
    let mapping_start = offset / DIRECT_READ_ALIGNMENT_BYTES * DIRECT_READ_ALIGNMENT_BYTES;
    let mapping_length = (offset + length - mapping_start).div_ceil(DIRECT_READ_ALIGNMENT_BYTES)
        * DIRECT_READ_ALIGNMENT_BYTES;
    // SAFETY: read-only shared mapping of a range inside the file, unmapped below.
    let mapping = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            mapping_length as usize,
            libc::PROT_READ,
            libc::MAP_SHARED,
            file.as_raw_fd(),
            mapping_start as libc::off_t,
        )
    };
    assert_ne!(
        mapping,
        libc::MAP_FAILED,
        "the residency mapping should succeed"
    );
    let mut page_states =
        vec![0 as libc::c_char; (mapping_length / DIRECT_READ_ALIGNMENT_BYTES) as usize];
    // SAFETY: one state byte per page of the mapping.
    let status =
        unsafe { libc::mincore(mapping, mapping_length as usize, page_states.as_mut_ptr()) };
    // SAFETY: unmapping exactly the mapping created above.
    unsafe { libc::munmap(mapping, mapping_length as usize) };
    assert_ne!(status, -1, "mincore should report residency");
    page_states.iter().any(|&page_state| page_state & 1 != 0)
}

/// One `preadv` reads a whole record and scatters it into `slot_destinations`,
/// in record order. A regular file returns a short read only at end of file,
/// which here means a truncated atlas, so a short read fails the probe.
pub(super) fn scatter_read_record(
    atlas_file: &File,
    record_offset: u64,
    slot_destinations: &mut [&mut [u8]],
) {
    let io_vectors: Vec<libc::iovec> = slot_destinations
        .iter_mut()
        .map(|slot_destination| libc::iovec {
            iov_base: slot_destination.as_mut_ptr().cast(),
            iov_len: slot_destination.len(),
        })
        .collect();
    let expected_byte_count: usize = slot_destinations
        .iter()
        .map(|slot_destination| slot_destination.len())
        .sum();
    // SAFETY: every iovec names one live, exclusively borrowed destination
    // slice for the duration of this call, sized exactly as declared.
    let read_byte_count = unsafe {
        libc::preadv(
            atlas_file.as_raw_fd(),
            io_vectors.as_ptr(),
            io_vectors.len() as i32,
            record_offset as libc::off_t,
        )
    };
    assert_eq!(
        read_byte_count,
        expected_byte_count as isize,
        "one atlas record should read whole: {}",
        std::io::Error::last_os_error()
    );
}

/// Bytes available to an unprivileged writer on the volume holding `path`.
pub(super) fn available_bytes_on_volume(path: &Path) -> u64 {
    let path_bytes = std::ffi::CString::new(path.as_os_str().as_encoded_bytes())
        .expect("the scratch path has no interior null");
    // SAFETY: statvfs fills a zeroed struct for a valid, null-terminated path.
    let mut volume_statistics: libc::statvfs = unsafe { std::mem::zeroed() };
    let status = unsafe { libc::statvfs(path_bytes.as_ptr(), &mut volume_statistics) };
    assert_eq!(status, 0, "statvfs should describe the scratch volume");
    volume_statistics.f_bavail as u64 * volume_statistics.f_frsize as u64
}
