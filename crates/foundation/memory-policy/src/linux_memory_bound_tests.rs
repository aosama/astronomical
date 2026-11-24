use std::path::{Path, PathBuf};

use super::{
    cgroup_limits_along_ancestors, cgroup_memory_limit_bytes, cgroup_v1_memory_group_path,
    cgroup_v1_memory_limit_bytes, cgroup_v2_group_path, memory_bound_bytes,
    physical_ram_bytes_from_linux_meminfo,
};

const GIBIBYTE: u64 = 1_073_741_824;

#[test]
fn should_read_the_physical_memory_size_from_linux_meminfo() {
    let meminfo_text = "MemTotal:       15727332 kB\nMemFree:         1048576 kB\n";
    assert_eq!(
        physical_ram_bytes_from_linux_meminfo(meminfo_text),
        Some(15_727_332 * 1024)
    );
}

#[test]
fn should_find_the_total_line_wherever_it_appears() {
    let meminfo_text = "MemFree:         1048576 kB\nMemTotal:       2 kB\n";
    assert_eq!(
        physical_ram_bytes_from_linux_meminfo(meminfo_text),
        Some(2048)
    );
}

#[test]
fn should_refuse_a_missing_zero_unitless_or_overflowing_total() {
    for unreadable_meminfo in [
        "",
        "MemFree: 1048576 kB\n",
        "MemTotal: 0 kB\n",
        "MemTotal: 1048576\n",
        "MemTotal: lots kB\n",
        "MemTotal: 18446744073709551615 kB\n",
    ] {
        assert_eq!(
            physical_ram_bytes_from_linux_meminfo(unreadable_meminfo),
            None,
            "{unreadable_meminfo:?} must not produce a ceiling"
        );
    }
}

#[test]
fn should_find_the_cgroup_v2_group_among_v1_hierarchies() {
    let membership_text = "12:memory:/legacy\n0::/user.slice/app.scope\n";
    assert_eq!(
        cgroup_v2_group_path(membership_text),
        Some(PathBuf::from("user.slice/app.scope"))
    );
}

#[test]
fn should_place_a_process_in_the_root_group_at_the_mount_itself() {
    assert_eq!(cgroup_v2_group_path("0::/\n"), Some(PathBuf::new()));
}

#[test]
fn should_find_no_group_outside_cgroup_v2() {
    assert_eq!(cgroup_v2_group_path("12:memory:/legacy\n"), None);
    assert_eq!(cgroup_v2_group_path(""), None);
}

#[test]
fn should_read_a_numeric_memory_max_as_a_limit() {
    assert_eq!(
        cgroup_memory_limit_bytes("17179869184\n"),
        Some(17_179_869_184)
    );
}

#[test]
fn should_treat_max_zero_and_garbage_as_no_limit() {
    for unlimited_text in ["max\n", "0\n", "", "lots\n"] {
        assert_eq!(
            cgroup_memory_limit_bytes(unlimited_text),
            None,
            "{unlimited_text:?} must not cap serving"
        );
    }
}

#[test]
fn should_take_the_tightest_of_physical_memory_and_every_cgroup_limit() {
    let sixty_four_gigabytes = 64 * 1_073_741_824;
    let sixteen_gigabytes = 16 * 1_073_741_824;
    let thirty_two_gigabytes = 32 * 1_073_741_824;
    assert_eq!(
        memory_bound_bytes(
            sixty_four_gigabytes,
            &[thirty_two_gigabytes, sixteen_gigabytes]
        ),
        sixteen_gigabytes
    );
}

#[test]
fn should_keep_physical_memory_without_a_tighter_cgroup_limit() {
    let eight_gigabytes = 8 * 1_073_741_824;
    assert_eq!(memory_bound_bytes(eight_gigabytes, &[]), eight_gigabytes);
    assert_eq!(
        memory_bound_bytes(eight_gigabytes, &[64 * 1_073_741_824]),
        eight_gigabytes
    );
}

/// Inside a cgroup namespace a process outside the namespace root sees a
/// climbing path; reading it would leave the cgroup mount.
#[test]
fn should_take_a_climbing_group_path_as_the_namespace_root() {
    assert_eq!(
        cgroup_v2_group_path("0::/../../other\n"),
        Some(PathBuf::new())
    );
}

#[test]
fn should_drop_the_deleted_suffix_of_a_removed_group() {
    assert_eq!(
        cgroup_v2_group_path("0::/system.slice/gone.service (deleted)\n"),
        Some(PathBuf::from("system.slice/gone.service"))
    );
}

#[test]
fn should_find_the_v1_memory_group_among_other_controllers() {
    let hybrid_membership = "12:cpu,cpuacct:/docker/abc\n4:memory:/docker/abc\n0::/init.scope\n";
    assert_eq!(
        cgroup_v1_memory_group_path(hybrid_membership),
        Some(PathBuf::from("docker/abc"))
    );
}

#[test]
fn should_find_no_v1_memory_group_on_a_pure_v2_host() {
    assert_eq!(cgroup_v1_memory_group_path("0::/user.slice\n"), None);
    assert_eq!(cgroup_v1_memory_group_path(""), None);
}

#[test]
fn should_read_a_v1_limit_and_treat_its_unlimited_sentinel_as_no_limit() {
    assert_eq!(
        cgroup_v1_memory_limit_bytes("17179869184\n"),
        Some(16 * GIBIBYTE)
    );
    assert_eq!(cgroup_v1_memory_limit_bytes("9223372036854771712\n"), None);
    assert_eq!(cgroup_v1_memory_limit_bytes("0\n"), None);
    assert_eq!(cgroup_v1_memory_limit_bytes("unlimited\n"), None);
}

fn write_limit(controller_root: &Path, group_path: &str, limit_file_name: &str, limit_text: &str) {
    let group_directory = controller_root.join(group_path);
    std::fs::create_dir_all(&group_directory).expect("the group folder should be created");
    std::fs::write(group_directory.join(limit_file_name), limit_text)
        .expect("the limit file should be written");
}

/// The walk reads the group and every ancestor: a tighter parent, a leaf
/// with no limit, and a missing middle file all count correctly.
#[test]
fn should_collect_every_finite_limit_from_the_group_up_to_the_root() {
    let cgroup_mount = tempfile::tempdir().expect("a temporary cgroup mount should exist");
    write_limit(
        cgroup_mount.path(),
        "user.slice/app.scope",
        "memory.max",
        "max\n",
    );
    write_limit(
        cgroup_mount.path(),
        "user.slice",
        "memory.max",
        "8589934592\n",
    );

    let cgroup_limits = cgroup_limits_along_ancestors(
        cgroup_mount.path(),
        Path::new("user.slice/app.scope/worker"),
        "memory.max",
        cgroup_memory_limit_bytes,
    );

    assert_eq!(cgroup_limits, [8 * GIBIBYTE]);
    assert_eq!(
        memory_bound_bytes(64 * GIBIBYTE, &cgroup_limits),
        8 * GIBIBYTE
    );
}

#[test]
fn should_read_the_root_group_and_find_no_limit_in_an_empty_mount() {
    let cgroup_mount = tempfile::tempdir().expect("a temporary cgroup mount should exist");

    assert!(
        cgroup_limits_along_ancestors(
            cgroup_mount.path(),
            Path::new(""),
            "memory.max",
            cgroup_memory_limit_bytes
        )
        .is_empty()
    );
    write_limit(
        cgroup_mount.path(),
        "",
        "memory.limit_in_bytes",
        "4294967296\n",
    );
    assert_eq!(
        cgroup_limits_along_ancestors(
            cgroup_mount.path(),
            Path::new(""),
            "memory.limit_in_bytes",
            cgroup_v1_memory_limit_bytes
        ),
        [4 * GIBIBYTE]
    );
}
