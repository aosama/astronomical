use super::{mac_hardware_model, macos_build_version, macos_product_version, system_tool_answer};

#[test]
fn should_read_a_dotted_macos_product_version() {
    let product_version = macos_product_version().expect("sw_vers answers on macOS");

    assert!(product_version.contains('.'));
    assert!(!product_version.contains(char::is_whitespace));
}

#[test]
fn should_read_a_macos_build_version_without_whitespace() {
    let build_version = macos_build_version().expect("sw_vers answers on macOS");

    assert!(!build_version.is_empty());
    assert!(!build_version.contains(char::is_whitespace));
}

#[test]
fn should_read_a_mac_hardware_model() {
    let hardware_model = mac_hardware_model().expect("sysctl answers on macOS");

    assert!(!hardware_model.is_empty());
    assert!(!hardware_model.contains('\n'));
}

#[test]
fn should_return_the_same_answer_every_time() {
    assert_eq!(macos_build_version(), macos_build_version());
}

#[test]
fn should_report_no_answer_for_a_tool_that_does_not_exist() {
    assert_eq!(
        system_tool_answer("/usr/bin/no-such-astronomical-tool", &[]),
        None
    );
}

#[test]
fn should_report_no_answer_for_a_tool_that_fails() {
    assert_eq!(system_tool_answer("/usr/bin/false", &[]), None);
}

#[test]
fn should_report_no_answer_for_a_tool_that_prints_nothing() {
    assert_eq!(system_tool_answer("/usr/bin/true", &[]), None);
}
