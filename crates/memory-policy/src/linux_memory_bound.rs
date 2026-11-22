//! The memory a Linux process may use: the machine's physical memory, lowered
//! by any cgroup v2 `memory.max` on the process's group or its ancestors.
//!
//! macOS bounds serving by `hw.memsize`, the whole machine (#1902). On Linux
//! the same rule needs the cgroup limit too: a container or a systemd unit can
//! cap a process far below the host's `MemTotal`, and planning past that cap
//! ends with the cgroup's OOM killer stopping the worker mid-request instead
//! of admission refusing the model. No reserve below the bound is kept, so the
//! policy stays the one macOS follows.
//!
//! The parsers are pure, so every rule is tested on any platform; only
//! [`read_linux_memory_bound_bytes`] touches `/proc` and `/sys`.

use std::path::{Path, PathBuf};

const LINUX_MEMINFO_PATH: &str = "/proc/meminfo";
const CGROUP_MEMBERSHIP_PATH: &str = "/proc/self/cgroup";
const CGROUP_V2_MOUNT_PATH: &str = "/sys/fs/cgroup";

/// The physical memory size from the `MemTotal:` line of `/proc/meminfo`
/// (reported in KiB). `None` when the line is missing or malformed, so a
/// caller never computes a ceiling from a guessed size.
#[must_use]
pub fn physical_ram_bytes_from_linux_meminfo(meminfo_text: &str) -> Option<u64> {
    for meminfo_line in meminfo_text.lines() {
        let Some(total_text) = meminfo_line.strip_prefix("MemTotal:") else {
            continue;
        };
        let mut total_fields = total_text.split_whitespace();
        let total_kibibytes = total_fields.next()?.parse::<u64>().ok()?;
        if total_fields.next() != Some("kB") || total_kibibytes == 0 {
            return None;
        }
        return total_kibibytes.checked_mul(1024);
    }
    None
}

/// The process's cgroup v2 group, from the `0::` line of `/proc/self/cgroup`,
/// as a path below the cgroup mount. `None` outside cgroup v2.
#[must_use]
pub fn cgroup_v2_group_path(cgroup_membership_text: &str) -> Option<PathBuf> {
    for membership_line in cgroup_membership_text.lines() {
        if let Some(group_path) = membership_line.strip_prefix("0::") {
            let relative_group_path = group_path.trim().trim_start_matches('/');
            return Some(PathBuf::from(relative_group_path));
        }
    }
    None
}

/// A `memory.max` file's limit in bytes. `max` means no limit, and anything
/// unreadable is treated as no limit rather than as a zero-byte cap.
#[must_use]
pub fn cgroup_memory_limit_bytes(memory_max_text: &str) -> Option<u64> {
    let limit_text = memory_max_text.trim();
    if limit_text == "max" {
        return None;
    }
    limit_text
        .parse::<u64>()
        .ok()
        .filter(|limit_bytes| *limit_bytes > 0)
}

/// The physical memory lowered by the tightest cgroup limit found.
#[must_use]
pub fn memory_bound_bytes(physical_ram_bytes: u64, cgroup_limits_bytes: &[u64]) -> u64 {
    let mut memory_bound_bytes = physical_ram_bytes;
    for cgroup_limit_bytes in cgroup_limits_bytes {
        memory_bound_bytes = memory_bound_bytes.min(*cgroup_limit_bytes);
    }
    memory_bound_bytes
}

/// Reads the Linux memory bound: `MemTotal`, lowered by every `memory.max`
/// from the process's cgroup up to the root.
///
/// # Errors
///
/// Returns a description when `/proc/meminfo` is unreadable or has no
/// readable `MemTotal` line. Missing cgroup files only mean "no limit".
pub fn read_linux_memory_bound_bytes() -> Result<u64, String> {
    let meminfo_text = std::fs::read_to_string(LINUX_MEMINFO_PATH)
        .map_err(|read_error| format!("failed to read {LINUX_MEMINFO_PATH}: {read_error}"))?;
    let physical_ram_bytes = physical_ram_bytes_from_linux_meminfo(&meminfo_text)
        .ok_or_else(|| format!("{LINUX_MEMINFO_PATH} has no readable MemTotal line"))?;
    Ok(memory_bound_bytes(
        physical_ram_bytes,
        &read_cgroup_limits_bytes(),
    ))
}

/// Every limit on the way from the process's group to the cgroup root. The
/// root has no `memory.max`, and a group without one simply adds no limit.
fn read_cgroup_limits_bytes() -> Vec<u64> {
    let Ok(cgroup_membership_text) = std::fs::read_to_string(CGROUP_MEMBERSHIP_PATH) else {
        return Vec::new();
    };
    let Some(group_path) = cgroup_v2_group_path(&cgroup_membership_text) else {
        return Vec::new();
    };
    let mut cgroup_limits_bytes = Vec::new();
    let mut ancestor_path = Some(group_path.as_path());
    while let Some(group_relative_path) = ancestor_path {
        let memory_max_path = Path::new(CGROUP_V2_MOUNT_PATH)
            .join(group_relative_path)
            .join("memory.max");
        if let Ok(memory_max_text) = std::fs::read_to_string(memory_max_path)
            && let Some(cgroup_limit_bytes) = cgroup_memory_limit_bytes(&memory_max_text)
        {
            cgroup_limits_bytes.push(cgroup_limit_bytes);
        }
        ancestor_path = group_relative_path.parent();
    }
    cgroup_limits_bytes
}

#[cfg(test)]
#[path = "linux_memory_bound_tests.rs"]
mod tests;
