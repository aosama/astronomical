//! Reusable build folders, so the native compile always runs at the same
//! absolute paths.
//!
//! The CMake compile commands embed absolute paths: the build folder, the
//! MLX sources FetchContent unpacks inside it, every include folder and
//! object file. sccache keys C and C++ compiles on those commands, so a
//! build in a fresh folder per invocation missed it on nearly every file; on
//! a four-core Linux laptop a rebuild took 725 seconds that way and 13 seconds
//! in a slot, with every C and C++ compile served by sccache. A small pool of
//! fixed folders keeps the paths identical between builds, including builds
//! of a different identity.
//!
//! Each slot is guarded by its own lock file, so concurrent builds take
//! different slots, and a slot that cannot be used (for example one left
//! with files the user cannot remove) is skipped rather than failing the
//! build. The lock stays open in the build's child processes, so a slot is
//! not handed out again while compilers a killed build started are still
//! writing into it. A slot is emptied before and after every build: nothing
//! from an earlier build is reused except through sccache, which keys on
//! the full compile, and no build tree stays on disk.

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

/// A held slot: its lock stays taken until this value is dropped, and
/// dropping it empties the slot.
pub(crate) struct HeldBuildSlot {
    pub(crate) cmake_build_directory: PathBuf,
    _slot_lock: File,
}

impl Drop for HeldBuildSlot {
    fn drop(&mut self) {
        if let Err(cleanup_error) = empty_directory(&self.cmake_build_directory) {
            eprintln!(
                "[native-build-store] status=cleanup-error path={} reason={cleanup_error}",
                self.cmake_build_directory.display()
            );
        }
    }
}

/// What one pass over the slots found.
enum SlotSearch {
    Taken(HeldBuildSlot),
    AllBusy { busy_slot_count: usize },
}

/// Takes the first usable free slot under `build_slots_directory`, waiting
/// with progress lines while all are busy, and returns it with an empty
/// CMake folder. Fails only when no slot can be used at all.
pub(crate) fn acquire_build_slot(
    build_slots_directory: &Path,
) -> Result<HeldBuildSlot, Box<dyn Error>> {
    create_real_directory(build_slots_directory)?;
    let wait_started_at = Instant::now();
    let mut next_progress_at = LOCK_PROGRESS_INTERVAL;
    loop {
        let busy_slot_count = match search_slots(build_slots_directory)? {
            SlotSearch::Taken(held_slot) => return Ok(held_slot),
            SlotSearch::AllBusy { busy_slot_count } => busy_slot_count,
        };
        if wait_started_at.elapsed() >= next_progress_at {
            // No ETA: how long the holders still need is not knowable here.
            eprintln!(
                "[native-build-store] status=waiting-for-build-slot busy_slots={busy_slot_count}/{BUILD_SLOT_COUNT} elapsed_seconds={} eta=unknown",
                wait_started_at.elapsed().as_secs()
            );
            next_progress_at += LOCK_PROGRESS_INTERVAL;
        }
        thread::sleep(LOCK_RETRY_INTERVAL);
    }
}

/// One pass: the first slot that locks and empties is taken; an unusable
/// slot is logged and skipped. An error means every slot was unusable.
fn search_slots(build_slots_directory: &Path) -> Result<SlotSearch, Box<dyn Error>> {
    let mut busy_slot_count = 0;
    let mut last_slot_error = None;
    for slot_index in 0..BUILD_SLOT_COUNT {
        match try_take_slot(build_slots_directory, slot_index) {
            Ok(Some(held_slot)) => return Ok(SlotSearch::Taken(held_slot)),
            Ok(None) => busy_slot_count += 1,
            Err(slot_error) => {
                eprintln!(
                    "[native-build-store] status=build-slot-unusable slot={slot_index} reason={slot_error}"
                );
                last_slot_error = Some(slot_error);
            }
        }
    }
    match last_slot_error {
        Some(slot_error) if busy_slot_count == 0 => Err(format!(
            "no native build slot under {} is usable; the last one failed with: {slot_error}",
            build_slots_directory.display()
        )
        .into()),
        _ => Ok(SlotSearch::AllBusy { busy_slot_count }),
    }
}

/// `None` when another build holds the slot.
fn try_take_slot(
    build_slots_directory: &Path,
    slot_index: usize,
) -> Result<Option<HeldBuildSlot>, Box<dyn Error>> {
    let lock_file_path = build_slots_directory.join(format!("slot-{slot_index}.lock"));
    refuse_symlink(&lock_file_path)?;
    let slot_lock = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(&lock_file_path)?;
    match fs4::FileExt::try_lock(&slot_lock) {
        Ok(()) => {}
        Err(TryLockError::WouldBlock) => return Ok(None),
        Err(TryLockError::Error(lock_error)) => return Err(lock_error.into()),
    }
    // Child processes inherit the open lock, so the slot stays locked until
    // the last compiler this build started exits, even if the build itself
    // is killed first.
    rustix::io::fcntl_setfd(&slot_lock, rustix::io::FdFlags::empty())?;
    let cmake_build_directory = build_slots_directory.join(format!("slot-{slot_index}"));
    empty_directory(&cmake_build_directory)?;
    Ok(Some(HeldBuildSlot {
        cmake_build_directory,
        _slot_lock: slot_lock,
    }))
}

/// Creates the slots folder, refusing a symlink so a tampered store cannot
/// point slot cleanup at a folder elsewhere on the machine.
fn create_real_directory(build_slots_directory: &Path) -> Result<(), Box<dyn Error>> {
    refuse_symlink(build_slots_directory)?;
    fs::create_dir_all(build_slots_directory)?;
    if !fs::symlink_metadata(build_slots_directory)?.is_dir() {
        return Err(format!("{} is not a folder", build_slots_directory.display()).into());
    }
    Ok(())
}

fn refuse_symlink(store_path: &Path) -> Result<(), Box<dyn Error>> {
    match fs::symlink_metadata(store_path) {
        Ok(path_metadata) if path_metadata.is_symlink() => Err(format!(
            "{} is a symlink, which the native store never follows",
            store_path.display()
        )
        .into()),
        _ => Ok(()),
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
