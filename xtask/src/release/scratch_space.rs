//! Scratch directories and mounted volumes that clean themselves up.
//!
//! The shell scripts relied on an exit `trap` to detach a mounted image and
//! remove their work directory; a failure between two steps could still
//! leave a volume mounted if the trap itself failed. Here both are guards
//! whose `Drop` runs on every exit path, including an early `?` return.

use std::path::{Path, PathBuf};
use std::process::Command;

/// A uniquely named directory under the system temporary directory,
/// removed when the guard drops.
pub(crate) struct ScratchDirectory {
    pub(crate) directory_path: PathBuf,
}

impl ScratchDirectory {
    pub(crate) fn create(name_prefix: &str) -> Result<Self, String> {
        let nanoseconds = match std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
            Ok(since_epoch) => since_epoch.subsec_nanos(),
            Err(_) => 0,
        };
        let directory_path = std::env::temp_dir().join(format!(
            "{name_prefix}.{}.{nanoseconds}",
            std::process::id()
        ));
        match std::fs::create_dir_all(&directory_path) {
            Ok(()) => Ok(Self { directory_path }),
            Err(create_error) => Err(format!(
                "could not create {}: {create_error}",
                directory_path.display()
            )),
        }
    }
}

impl Drop for ScratchDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.directory_path);
    }
}

/// A disk image attached at a mount point, force-detached when the guard
/// drops unless it was already detached cleanly.
pub(crate) struct MountedVolume {
    pub(crate) mount_point: PathBuf,
    is_attached: bool,
}

impl MountedVolume {
    pub(crate) fn attach(
        disk_image_path: &Path,
        mount_point: &Path,
        attach_flags: &[&str],
    ) -> Result<Self, String> {
        let attach_status = Command::new("hdiutil")
            .arg("attach")
            .args(attach_flags)
            .arg("-mountpoint")
            .arg(mount_point)
            .arg(disk_image_path)
            .stdout(std::process::Stdio::null())
            .status();
        match attach_status {
            Ok(exit_status) if exit_status.success() => Ok(Self {
                mount_point: mount_point.to_path_buf(),
                is_attached: true,
            }),
            _ => Err(format!("could not attach {}", disk_image_path.display())),
        }
    }

    pub(crate) fn detach(mut self) -> Result<(), String> {
        let detach_status = Command::new("hdiutil")
            .arg("detach")
            .arg(&self.mount_point)
            .status();
        match detach_status {
            Ok(exit_status) if exit_status.success() => {
                self.is_attached = false;
                Ok(())
            }
            _ => Err(format!("could not detach {}", self.mount_point.display())),
        }
    }
}

impl Drop for MountedVolume {
    fn drop(&mut self) {
        if self.is_attached {
            let _ = Command::new("hdiutil")
                .arg("detach")
                .arg(&self.mount_point)
                .arg("-force")
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status();
        }
    }
}
