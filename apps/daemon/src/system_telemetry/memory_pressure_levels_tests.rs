use super::{parse_linux_memory_pressure_level, parse_macos_memory_pressure_level};

const CALM_PRESSURE_STALL: &str = "some avg10=0.00 avg60=0.00 avg300=0.00 total=0
full avg10=0.00 avg60=0.00 avg300=0.00 total=0
";

#[test]
fn should_read_calm_linux_pressure_as_normal() {
    assert_eq!(
        parse_linux_memory_pressure_level(CALM_PRESSURE_STALL),
        Some("normal")
    );
}

#[test]
fn should_read_linux_partial_stalls_at_the_threshold_as_a_warning() {
    let partial_stall = "some avg10=10.00 avg60=4.00 avg300=1.00 total=99
full avg10=9.99 avg60=1.00 avg300=0.10 total=10
";
    assert_eq!(
        parse_linux_memory_pressure_level(partial_stall),
        Some("warning")
    );
}

#[test]
fn should_read_linux_partial_stalls_just_below_the_threshold_as_normal() {
    let light_stall = "some avg10=9.99 avg60=4.00 avg300=1.00 total=99
full avg10=0.50 avg60=0.10 avg300=0.00 total=10
";
    assert_eq!(
        parse_linux_memory_pressure_level(light_stall),
        Some("normal")
    );
}

#[test]
fn should_read_linux_full_stalls_at_the_threshold_as_critical() {
    let full_stall = "some avg10=42.00 avg60=20.00 avg300=5.00 total=999
full avg10=10.00 avg60=6.00 avg300=1.00 total=500
";
    assert_eq!(
        parse_linux_memory_pressure_level(full_stall),
        Some("critical")
    );
}

#[test]
fn should_report_unavailable_for_malformed_or_partial_linux_pressure() {
    for unreadable_pressure in [
        "",
        "some avg10=1.00 avg60=0.00 avg300=0.00 total=0\n",
        "some avg10=abc avg60=0.00\nfull avg10=0.00 avg60=0.00\n",
        "some avg10=101.00\nfull avg10=0.00\n",
        "some avg60=1.00\nfull avg10=0.00\n",
    ] {
        assert_eq!(
            parse_linux_memory_pressure_level(unreadable_pressure),
            None,
            "{unreadable_pressure:?} must read as unavailable, not as normal"
        );
    }
}

#[test]
fn should_keep_mapping_the_macos_sysctl_bitmask_on_every_platform() {
    assert_eq!(parse_macos_memory_pressure_level("1\n"), Some("normal"));
    assert_eq!(parse_macos_memory_pressure_level("2"), Some("warning"));
    assert_eq!(parse_macos_memory_pressure_level("4"), Some("critical"));
    assert_eq!(parse_macos_memory_pressure_level("0"), None);
}
