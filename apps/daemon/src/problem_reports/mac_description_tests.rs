use super::MacDescription;

#[test]
fn should_read_chip_memory_and_macos_version_from_sysctl_values() {
    let mac = MacDescription::from_sysctl_values("Apple M5 Pro\n51539607552\n27.2\n")
        .expect("three valid values should parse");
    assert_eq!(mac.chip_name, "Apple M5 Pro");
    assert_eq!(mac.memory_gb, 48);
    assert_eq!(mac.macos_version, "27.2");
    assert_eq!(mac.display_text(), "Apple M5 Pro, 48 GB, macOS 27.2");
}

#[test]
fn should_round_memory_to_the_nearest_whole_gb() {
    let just_under_eight = MacDescription::from_sysctl_values("Apple M1\n8589000000\n15.0\n")
        .expect("values should parse");
    assert_eq!(just_under_eight.memory_gb, 8);
}

#[test]
fn should_refuse_missing_empty_or_unreadable_values() {
    for sysctl_values_text in [
        "",
        "Apple M5 Pro\n",
        "Apple M5 Pro\nlots\n27.2\n",
        "Apple M5 Pro\n0\n27.2\n",
        "\n51539607552\n27.2\n",
        "Apple M5 Pro\n51539607552\n\n",
    ] {
        assert_eq!(
            MacDescription::from_sysctl_values(sysctl_values_text),
            None,
            "{sysctl_values_text:?}"
        );
    }
}
