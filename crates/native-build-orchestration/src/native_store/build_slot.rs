//! Reusable build folders, so the native compile always runs at the same
//! absolute paths.
//!
//! The CMake compile commands embed absolute paths: the build folder, the
//! MLX sources FetchContent unpacks inside it, every include folder and
//! object file. sccache keys C and C++ compiles on those commands, so a
//! build in a fresh folder per invocation missed it on nearly every file
//! (436 of 504 compiles on one Linux rebuild, roughly twelve minutes). A
//! small pool of fixed folders keeps the paths identical between builds,
//! including builds of a different identity, and lets sccache serve every
//! object whose source and flags did not change.
//!
//! Each slot is guarded by its own lock file, so concurrent builds of
//! different identities take different slots. A slot's CMake folder is
//! emptied before every build, so nothing from an earlier build is reused
//! except through sccache, which keys on the full compile.

use std::{
    error::Error,
    fs::{self, File, OpenOptions},
    path::{Path, PathBuf},
    thread,
    time::Instant,
};

use fs4::TryLockError;

use super::{LOCK_PROGRESS_INTERVAL, LOCK_RETRY_INTERVAL};

/// Enough for the profiles and worktrees that build at once on one machine;
/// a further build waits for a slot rather than compiling at new paths.
pub(crate) const BUILD_SLOT_COUNT: usize = 4;

/// A held slot: its lock stays taken until this value is dropped.
pub(crate) struct HeldBuildSlot {
    pub(crate) cmake_build_directory: PathBuf,
    _slot_lock: File,
}

/// Takes the first free slot under `build_slots_directory`, waiting with
/// progress lines when all are busy, and returns it with an empty CMake
/// folder.
pub(crate) fn acquire_build_slot(
    build_slots_directory: &Path,
) -> Result<HeldBuildSlot, Box<dyn Error>> {
    fs::create_dir_all(build_slots_directory)?;
    let wait_started_at = Instant::now();
    let mut next_progress_at = LOCK_PROGRESS_INTERVAL;
    loop {
        for slot_index in 0..BUILD_SLOT_COUNT {
            if let Some(slot_lock) = try_lock_slot(build_slots_directory, slot_index)? {
                let cmake_build_directory =
                    build_slots_directory.join(format!("slot-{slot_index}"));
                empty_directory(&cmake_build_directory)?;
                return Ok(HeldBuildSlot {
                    cmake_build_directory,
                    _slot_lock: slot_lock,
                });
            }
        }
        if wait_started_at.elapsed() >= next_progress_at {
            eprintln!(
                "[native-build-store] status=waiting-for-build-slot elapsed_seconds={}",
                wait_started_at.elapsed().as_secs()
            );
            next_progress_at += LOCK_PROGRESS_INTERVAL;
        }
        thread::sleep(LOCK_RETRY_INTERVAL);
    }
}

fn try_lock_slot(
    build_slots_directory: &Path,
    slot_index: usize,
) -> Result<Option<File>, Box<dyn Error>> {
    let slot_lock = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(build_slots_directory.join(format!("slot-{slot_index}.lock")))?;
    match fs4::FileExt::try_lock(&slot_lock) {
        Ok(()) => Ok(Some(slot_lock)),
        Err(TryLockError::WouldBlock) => Ok(None),
        Err(TryLockError::Error(lock_error)) => Err(lock_error.into()),
    }
}

/// Removes whatever an earlier build left, without following a symlink at
/// the slot path, and recreates the folder empty.
fn empty_directory(slot_directory: &Path) -> Result<(), Box<dyn Error>> {
    match fs::symlink_metadata(slot_directory) {
        Ok(slot_metadata) if slot_metadata.is_dir() => fs::remove_dir_all(slot_directory)?,
        Ok(_) => fs::remove_file(slot_directory)?,
        Err(metadata_error) if metadata_error.kind() == std::io::ErrorKind::NotFound => {}
        Err(metadata_error) => return Err(metadata_error.into()),
    }
    fs::create_dir(slot_directory)?;
    Ok(())
}

#[cfg(test)]
mod tests;
