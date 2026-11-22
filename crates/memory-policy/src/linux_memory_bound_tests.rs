use std::path::PathBuf;

use super::{
    cgroup_memory_limit_bytes, cgroup_v2_group_path, memory_bound_bytes,
    physical_ram_bytes_from_linux_meminfo,
};

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
