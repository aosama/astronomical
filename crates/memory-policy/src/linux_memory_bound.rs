//! The memory a Linux process may use: the machine's physical memory, lowered
//! by any cgroup memory limit on the process's group or its ancestors: v2's
//! `memory.max`, and on cgroup v1 or hybrid hosts (older Docker and systemd
//! setups) the v1 memory controller's `memory.limit_in_bytes`.
//!
//! macOS bounds serving by `hw.memsize`, the whole machine (#1902). On Linux
//! the same rule needs the cgroup limit too: a container or a systemd unit can
//! cap a process far below the host's `MemTotal`, and planning past that cap
//! ends with the cgroup's OOM killer stopping the worker mid-request instead
//! of admission refusing the model. No reserve below the bound is kept, so the
//! policy stays the one macOS follows.
//!
//! The parsers are pure, and the ancestor walk takes its mount root as a
//! parameter, so every rule is tested on any platform; only
//! [`read_linux_memory_bound_bytes`] reads the real `/proc` and `/sys`.

use std::path::{Path, PathBuf};

const LINUX_MEMINFO_PATH: &str = "/proc/meminfo";
const CGROUP_MEMBERSHIP_PATH: &str = "/proc/self/cgroup";
const CGROUP_MOUNT_PATH: &str = "/sys/fs/cgroup";
/// Where a v1 or hybrid host mounts the v1 memory controller, below the
/// cgroup mount.
const CGROUP_V1_MEMORY_CONTROLLER_DIRECTORY: &str = "memory";
const CGROUP_V2_LIMIT_FILE_NAME: &str = "memory.max";
const CGROUP_V1_LIMIT_FILE_NAME: &str = "memory.limit_in_bytes";
/// cgroup v1 reports "no limit" as the page counter's maximum, a number near
/// 2^63 that varies with the page size; anything this large is no limit.
const CGROUP_V1_UNLIMITED_FLOOR_BYTES: u64 = 1 << 62;

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
            return Some(confined_group_path(group_path));
        }
    }
    None
}

/// The process's group in the cgroup v1 memory controller, from a
/// `N:…memory…:/path` line, as a path below that controller's mount. `None`
/// when no v1 hierarchy carries the memory controller.
#[must_use]
pub fn cgroup_v1_memory_group_path(cgroup_membership_text: &str) -> Option<PathBuf> {
    for membership_line in cgroup_membership_text.lines() {
        let mut membership_fields = membership_line.splitn(3, ':');
        let (Some(_hierarchy_id), Some(controllers), Some(group_path)) = (
            membership_fields.next(),
            membership_fields.next(),
            membership_fields.next(),
        ) else {
            continue;
        };
        if controllers.split(',').any(is_memory_controller) {
            return Some(confined_group_path(group_path));
        }
    }
    None
}

fn is_memory_controller(controller_name: &str) -> bool {
    controller_name == "memory"
}

/// A group path made relative to the mount. Inside a cgroup namespace a
/// process outside the namespace's root sees a path such as `/../../other`;
/// joining that to the mount would read files outside it, so any path that
/// climbs is taken as the namespace's root group instead. The kernel's
/// ` (deleted)` suffix for a removed group is dropped.
fn confined_group_path(group_path_text: &str) -> PathBuf {
    let group_path_text = group_path_text.trim();
    let group_path_text = group_path_text
        .strip_suffix(" (deleted)")
        .unwrap_or(group_path_text);
    let relative_group_path = PathBuf::from(group_path_text.trim_start_matches('/'));
    if relative_group_path.components().any(is_parent_directory) {
        return PathBuf::new();
    }
    relative_group_path
}

fn is_parent_directory(path_component: std::path::Component<'_>) -> bool {
    path_component == std::path::Component::ParentDir
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

/// A v1 `memory.limit_in_bytes` file's limit in bytes. The kernel's
/// "unlimited" sentinel, zero and anything unreadable mean no limit.
#[must_use]
pub fn cgroup_v1_memory_limit_bytes(limit_in_bytes_text: &str) -> Option<u64> {
    limit_in_bytes_text
        .trim()
        .parse::<u64>()
        .ok()
        .filter(is_finite_v1_limit)
}

fn is_finite_v1_limit(limit_bytes: &u64) -> bool {
    *limit_bytes > 0 && *limit_bytes < CGROUP_V1_UNLIMITED_FLOOR_BYTES
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

/// Every limit on the way from the process's group to the cgroup root, in
/// both the v2 hierarchy and the v1 memory controller. A host uses one or the
/// other for memory, so at most one of the walks finds limits.
fn read_cgroup_limits_bytes() -> Vec<u64> {
    let Ok(cgroup_membership_text) = std::fs::read_to_string(CGROUP_MEMBERSHIP_PATH) else {
        return Vec::new();
    };
    let cgroup_mount = Path::new(CGROUP_MOUNT_PATH);
    let mut cgroup_limits_bytes = Vec::new();
    if let Some(group_path) = cgroup_v2_group_path(&cgroup_membership_text) {
        cgroup_limits_bytes.extend(cgroup_limits_along_ancestors(
            cgroup_mount,
            &group_path,
            CGROUP_V2_LIMIT_FILE_NAME,
            cgroup_memory_limit_bytes,
        ));
    }
    if let Some(group_path) = cgroup_v1_memory_group_path(&cgroup_membership_text) {
        cgroup_limits_bytes.extend(cgroup_limits_along_ancestors(
            &cgroup_mount.join(CGROUP_V1_MEMORY_CONTROLLER_DIRECTORY),
            &group_path,
            CGROUP_V1_LIMIT_FILE_NAME,
            cgroup_v1_memory_limit_bytes,
        ));
    }
    cgroup_limits_bytes
}

/// Reads `limit_file_name` in `group_path` and each ancestor up to and
/// including `controller_root`, keeping every value `parse_limit_bytes`
/// accepts. A group without the file, or with an unlimited value, adds no
/// limit.
#[must_use]
pub fn cgroup_limits_along_ancestors(
    controller_root: &Path,
    group_path: &Path,
    limit_file_name: &str,
    parse_limit_bytes: fn(&str) -> Option<u64>,
) -> Vec<u64> {
    let mut cgroup_limits_bytes = Vec::new();
    let mut ancestor_path = Some(group_path);
    while let Some(group_relative_path) = ancestor_path {
        let limit_file_path = controller_root
            .join(group_relative_path)
            .join(limit_file_name);
        if let Ok(limit_text) = std::fs::read_to_string(limit_file_path)
            && let Some(cgroup_limit_bytes) = parse_limit_bytes(&limit_text)
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
