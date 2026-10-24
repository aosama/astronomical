//! Keeps expert-paging reads out of the operating system's file cache.
//!
//! # Why expert pages bypass the unified buffer cache
//!
//! Solid-state-drive expert paging exists because the model does not fit in
//! memory. Every routed expert page read through the ordinary file path also
//! leaves a copy in the macOS unified buffer cache, so streaming a large
//! mixture-of-experts shard fills the cache with bytes the runtime already
//! copied into its own MLX arrays. That duplicate copy competes with the
//! resident model weights and the key-value cache for the same physical memory
//! and pushes the system toward compression and swap — the exact pressure
//! paging was built to avoid. `fcntl(F_NOCACHE)` asks the kernel to serve these
//! reads without retaining their pages, so the runtime's own residency policy
//! stays the single owner of which expert bytes live in memory.
//!
//! The trade-off is deliberate: an expert page that is re-read soon after its
//! eviction now comes from the drive instead of from the file cache. The
//! runtime's resident expert set, not the kernel's cache, is the intended place
//! to keep hot experts.

use std::{fs::File, io};

/// Turns off unified-buffer-cache retention for every read on this descriptor.
///
/// The setting belongs to the open file description, so it must be applied to
/// the exact descriptor the bounded reader keeps for deferred MLX reads. On
/// operating systems without `F_NOCACHE` this is a no-op.
pub(crate) fn disable_file_cache_retention(source_file: &File) -> io::Result<()> {
    disable_file_cache_retention_for_platform(source_file)
}

#[cfg(target_os = "macos")]
fn disable_file_cache_retention_for_platform(source_file: &File) -> io::Result<()> {
    use std::os::fd::AsRawFd;

    // SAFETY: `source_file` owns an open descriptor for the duration of this
    // call, and F_NOCACHE takes a plain integer argument with no pointers.
    let fcntl_status = unsafe { libc::fcntl(source_file.as_raw_fd(), libc::F_NOCACHE, 1) };
    if fcntl_status == -1 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn disable_file_cache_retention_for_platform(_source_file: &File) -> io::Result<()> {
    Ok(())
}

#[cfg(test)]
mod page_cache_bypass_tests {
    use std::{
        io::Write,
        os::unix::fs::FileExt,
        process::{Command, Stdio},
    };

    use super::disable_file_cache_retention;

    const EXPERT_PAGE_FIXTURE_BYTES: &[u8] =
        b"Two households, both alike in dignity, in fair Verona, where we lay our scene";

    #[test]
    fn should_read_identical_bytes_after_disabling_file_cache_retention() {
        let mut expert_shard_file = tempfile::tempfile().expect("temporary shard should open");
        expert_shard_file
            .write_all(EXPERT_PAGE_FIXTURE_BYTES)
            .expect("fixture bytes should write");

        disable_file_cache_retention(&expert_shard_file)
            .expect("a regular file should accept cache bypass");

        let mut read_back_bytes = vec![0_u8; EXPERT_PAGE_FIXTURE_BYTES.len()];
        expert_shard_file
            .read_exact_at(&mut read_back_bytes, 0)
            .expect("positional read should succeed after cache bypass");
        assert_eq!(read_back_bytes, EXPERT_PAGE_FIXTURE_BYTES);
    }

    #[test]
    fn should_accept_cache_bypass_on_an_empty_file() {
        let empty_shard_file = tempfile::tempfile().expect("temporary shard should open");

        disable_file_cache_retention(&empty_shard_file)
            .expect("an empty regular file should accept cache bypass");
    }

    #[test]
    fn should_accept_repeated_cache_bypass_on_the_same_descriptor() {
        let expert_shard_file = tempfile::tempfile().expect("temporary shard should open");

        disable_file_cache_retention(&expert_shard_file).expect("first bypass should succeed");
        disable_file_cache_retention(&expert_shard_file)
            .expect("repeating bypass on one descriptor should be idempotent");
    }

    #[test]
    fn should_not_fail_partial_reads_past_the_end_of_a_bypassed_file() {
        let mut expert_shard_file = tempfile::tempfile().expect("temporary shard should open");
        expert_shard_file
            .write_all(EXPERT_PAGE_FIXTURE_BYTES)
            .expect("fixture bytes should write");
        disable_file_cache_retention(&expert_shard_file).expect("bypass should succeed");

        let mut overrun_destination = vec![0_u8; EXPERT_PAGE_FIXTURE_BYTES.len() + 1];
        let short_read_error = expert_shard_file
            .read_exact_at(&mut overrun_destination, 0)
            .expect_err("reading past the end must still report a short read");
        assert_eq!(short_read_error.kind(), std::io::ErrorKind::UnexpectedEof);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn should_report_an_error_when_the_descriptor_is_not_a_cacheable_file() {
        // A pipe has no unified-buffer-cache pages, so macOS rejects F_NOCACHE
        // on it; the error must surface instead of being silently swallowed.
        let mut pipe_producer = Command::new("/bin/echo")
            .stdout(Stdio::piped())
            .spawn()
            .expect("pipe producer should start");
        let pipe_reader = std::fs::File::from(std::os::fd::OwnedFd::from(
            pipe_producer.stdout.take().expect("pipe should exist"),
        ));

        let bypass_outcome = disable_file_cache_retention(&pipe_reader);
        pipe_producer.wait().expect("pipe producer should exit");

        assert!(
            bypass_outcome.is_err(),
            "cache bypass on a pipe should fail, got {bypass_outcome:?}"
        );
    }
}
