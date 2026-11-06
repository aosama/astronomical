//! What this Mac and its macOS are, each read once per process.
//!
//! Every answer starts a system tool, and a program set asks once per program, so
//! the answers are kept. `None` means the tool could not be run or said nothing.

use std::sync::OnceLock;

const SW_VERS_PATH: &str = "/usr/bin/sw_vers";
const SYSCTL_PATH: &str = "/usr/sbin/sysctl";

/// The macOS build and Mac a proof or a reading holds for. An update or a move to another
/// Mac, by a migration or a restored backup, can change what the Neural Engine does.
pub(crate) fn host_environment_of(
    macos_product_version: &str,
    macos_build_version: &str,
    mac_hardware_model: &str,
) -> String {
    format!("macOS {macos_product_version} ({macos_build_version}) on {mac_hardware_model}")
}

/// This process's environment, or `None` when the system would not describe itself, in
/// which case nothing that depends on it is recorded.
pub(crate) fn current_host_environment() -> Option<&'static str> {
    static HOST_ENVIRONMENT: OnceLock<Option<String>> = OnceLock::new();
    HOST_ENVIRONMENT
        .get_or_init(describe_current_host_environment)
        .as_deref()
}

fn describe_current_host_environment() -> Option<String> {
    Some(host_environment_of(
        macos_product_version()?,
        macos_build_version()?,
        mac_hardware_model()?,
    ))
}

/// For example `27.0.1`.
pub(crate) fn macos_product_version() -> Option<&'static str> {
    static PRODUCT_VERSION: OnceLock<Option<String>> = OnceLock::new();
    PRODUCT_VERSION
        .get_or_init(read_macos_product_version)
        .as_deref()
}

/// Changes with a supplemental update that keeps the product version.
pub(crate) fn macos_build_version() -> Option<&'static str> {
    static BUILD_VERSION: OnceLock<Option<String>> = OnceLock::new();
    BUILD_VERSION
        .get_or_init(read_macos_build_version)
        .as_deref()
}

/// Apple's model identifier for this Mac, which names its chip family.
pub(crate) fn mac_hardware_model() -> Option<&'static str> {
    static HARDWARE_MODEL: OnceLock<Option<String>> = OnceLock::new();
    HARDWARE_MODEL
        .get_or_init(read_mac_hardware_model)
        .as_deref()
}

fn read_macos_product_version() -> Option<String> {
    system_tool_answer(SW_VERS_PATH, &["-productVersion"])
}

fn read_macos_build_version() -> Option<String> {
    system_tool_answer(SW_VERS_PATH, &["-buildVersion"])
}

fn read_mac_hardware_model() -> Option<String> {
    system_tool_answer(SYSCTL_PATH, &["-n", "hw.model"])
}

fn system_tool_answer(tool_path: &str, arguments: &[&str]) -> Option<String> {
    let tool_output = std::process::Command::new(tool_path)
        .args(arguments)
        .output()
        .ok()?;
    let answer_text = String::from_utf8(tool_output.stdout).ok()?;
    let trimmed_answer = answer_text.trim();
    (tool_output.status.success() && !trimmed_answer.is_empty())
        .then_some(trimmed_answer.to_owned())
}

#[cfg(test)]
#[path = "host_description_tests.rs"]
mod tests;
