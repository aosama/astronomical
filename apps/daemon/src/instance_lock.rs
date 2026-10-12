use std::fs::{File, OpenOptions};
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};

use fs4::TryLockError;

/// Exclusive process-lifetime ownership of one instance's writable state.
pub struct AstronomicalInstanceLock {
    _locked_file: File,
}

impl AstronomicalInstanceLock {
    pub fn acquire(lock_file_path: &Path) -> Result<Self, AstronomicalInstanceLockError> {
        let state_directory = lock_file_path.parent().ok_or_else(|| {
            AstronomicalInstanceLockError::InvalidLockPath {
                lock_file_path: lock_file_path.to_owned(),
            }
        })?;
        std::fs::create_dir_all(state_directory).map_err(|source| {
            AstronomicalInstanceLockError::CreateStateDirectory {
                state_directory: state_directory.to_owned(),
                source,
            }
        })?;
        let locked_file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .mode(0o600)
            .open(lock_file_path)
            .map_err(|source| AstronomicalInstanceLockError::OpenLockFile {
                lock_file_path: lock_file_path.to_owned(),
                source,
            })?;
        match fs4::FileExt::try_lock(&locked_file) {
            Ok(()) => Ok(Self {
                _locked_file: locked_file,
            }),
            Err(TryLockError::WouldBlock) => Err(AstronomicalInstanceLockError::AlreadyRunning),
            Err(TryLockError::Error(source)) => {
                Err(AstronomicalInstanceLockError::AcquireLock { source })
            }
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum AstronomicalInstanceLockError {
    #[error("Astronomical is already running for the selected instance")]
    AlreadyRunning,
    #[error("instance lock path has no state directory: {lock_file_path:?}")]
    InvalidLockPath { lock_file_path: PathBuf },
    #[error("failed to create Astronomical instance state directory at {state_directory:?}")]
    CreateStateDirectory {
        state_directory: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to open Astronomical instance lock file at {lock_file_path:?}")]
    OpenLockFile {
        lock_file_path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to acquire Astronomical instance lock")]
    AcquireLock {
        #[source]
        source: std::io::Error,
    },
}
