//! Keeps a model's weight files out of the macOS file cache while a probe
//! serves, so expert reads cost what they cost on a Mac whose RAM is no
//! larger than the MLX limit.
//!
//! A probe's MLX limit caps MLX's own memory, not the file cache: on a Mac
//! with more RAM than the limit, the cache holds most of the model and
//! "storage" reads come from memory (measured: 96% of decode expert bytes on
//! a 48 GB Mac at a 16 GB limit). `msync(MS_INVALIDATE)` on a read-only
//! shared mapping drops a file's clean cached pages (measured with `mincore`:
//! a 0.86 GB layer file went from fully resident to none in 7 ms), so a
//! background thread that sweeps every weight file keeps the cache empty.
//! This is the pessimistic end of a small Mac, which still caches a little.

use std::fs::File;
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::Duration;

/// Pause between sweeps: short enough that a re-read expert is rarely still
/// cached when decode asks for it again, long enough that the sweeps do not
/// compete with the serving threads for a core.
const SWEEP_PAUSE: Duration = Duration::from_millis(10);

/// One weight file mapped read-only for invalidation.
struct MappedWeightFile {
    mapping_address: *mut libc::c_void,
    mapping_length: usize,
    _file: File,
}

// SAFETY: the mapping is read-only, never dereferenced, and used only by
// msync and munmap from the sweeping thread that owns it.
unsafe impl Send for MappedWeightFile {}

impl MappedWeightFile {
    fn map(path: &Path) -> Self {
        let file = File::open(path)
            .unwrap_or_else(|error| panic!("{} should open: {error}", path.display()));
        let mapping_length = usize::try_from(
            file.metadata()
                .unwrap_or_else(|error| panic!("{} should stat: {error}", path.display()))
                .len(),
        )
        .expect("the weight file fits the address space");
        // SAFETY: a fresh read-only shared mapping of a live descriptor; the
        // pages are never read through it.
        let mapping_address = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                mapping_length,
                libc::PROT_READ,
                libc::MAP_SHARED,
                file.as_raw_fd(),
                0,
            )
        };
        assert_ne!(
            mapping_address,
            libc::MAP_FAILED,
            "{} should map: {}",
            path.display(),
            std::io::Error::last_os_error()
        );
        Self {
            mapping_address,
            mapping_length,
            _file: file,
        }
    }

    fn invalidate_cached_pages(&self) {
        // SAFETY: the address and length are exactly this live mapping.
        let status = unsafe {
            libc::msync(
                self.mapping_address,
                self.mapping_length,
                libc::MS_INVALIDATE,
            )
        };
        assert_eq!(
            status,
            0,
            "msync invalidation should succeed: {}",
            std::io::Error::last_os_error()
        );
    }
}

impl Drop for MappedWeightFile {
    fn drop(&mut self) {
        // SAFETY: unmaps exactly the mapping created in `map`.
        unsafe { libc::munmap(self.mapping_address, self.mapping_length) };
    }
}

/// The running sweeper; `finish` stops it and returns its sweep count.
/// Dropping it also stops it, so a journey that times out or panics never
/// leaves the cache being evicted under the next serial test.
pub(super) struct ColdFileCache {
    should_stop: Arc<AtomicBool>,
    sweeping_thread: Option<JoinHandle<u64>>,
}

impl ColdFileCache {
    /// Evicts every `.safetensors` and `.atlas` file under `model_directory`
    /// once, then keeps sweeping until `finish`.
    pub(super) fn start(model_directory: &Path) -> Self {
        let weight_files: Vec<MappedWeightFile> = weight_file_paths(model_directory)
            .iter()
            .map(|path| MappedWeightFile::map(path))
            .collect();
        assert!(
            !weight_files.is_empty(),
            "{} holds no weight files to keep cold",
            model_directory.display()
        );
        for weight_file in &weight_files {
            weight_file.invalidate_cached_pages();
        }
        let should_stop = Arc::new(AtomicBool::new(false));
        let sweeping_thread = std::thread::spawn({
            let should_stop = Arc::clone(&should_stop);
            move || sweep_until_stopped(&weight_files, &should_stop)
        });
        Self {
            should_stop,
            sweeping_thread: Some(sweeping_thread),
        }
    }

    pub(super) fn finish(mut self) -> u64 {
        self.should_stop.store(true, Ordering::Relaxed);
        self.sweeping_thread
            .take()
            .expect("the sweeper runs until finished")
            .join()
            .expect("the file cache sweeper should not panic")
    }
}

impl Drop for ColdFileCache {
    fn drop(&mut self) {
        self.should_stop.store(true, Ordering::Relaxed);
        if let Some(sweeping_thread) = self.sweeping_thread.take() {
            // Joined so the mappings are gone before the next test starts;
            // a sweeper panic is already being reported by this unwind.
            let _ = sweeping_thread.join();
        }
    }
}

fn sweep_until_stopped(weight_files: &[MappedWeightFile], should_stop: &AtomicBool) -> u64 {
    let mut sweep_count = 0_u64;
    while !should_stop.load(Ordering::Relaxed) {
        for weight_file in weight_files {
            weight_file.invalidate_cached_pages();
        }
        sweep_count += 1;
        std::thread::sleep(SWEEP_PAUSE);
    }
    sweep_count
}

fn weight_file_paths(model_directory: &Path) -> Vec<PathBuf> {
    let mut weight_file_paths = Vec::new();
    let mut pending_directories = vec![model_directory.to_path_buf()];
    while let Some(directory) = pending_directories.pop() {
        for entry in std::fs::read_dir(&directory)
            .unwrap_or_else(|error| panic!("{} should list: {error}", directory.display()))
        {
            let path = entry.expect("directory entries should read").path();
            if path.is_dir() {
                pending_directories.push(path);
            } else if path
                .extension()
                .is_some_and(|extension| extension == "safetensors" || extension == "atlas")
            {
                weight_file_paths.push(path);
            }
        }
    }
    weight_file_paths.sort();
    weight_file_paths
}
