//! Finds lint settings that would let a long function through without any
//! Rust attribute: a Cargo manifest that does not deny `clippy::too_many_lines`
//! (the lint is allow-by-default, so a crate with its own lints table that
//! leaves it out is never checked), a manifest that sets it to anything but
//! `deny`, a `clippy.toml` beside a crate (clippy reads the nearest one, so it
//! could raise the threshold for that crate), and a root `clippy.toml` whose
//! threshold differs from the budget this scan enforces.
//!
//! The manifests are read as text, line by line: the repository writes each
//! lint on its own line, and an inline table such as `lints = { .. }` is not
//! used anywhere, so a full TOML parser would add a dependency for no case.

use super::{ExemptionKind, FUNCTION_LENGTH_BUDGET_CODE_LINES, FunctionLengthExemption};

const FUNCTION_LENGTH_LINT_NAME: &str = "too_many_lines";

/// Whether a file name is a Cargo manifest.
pub(crate) fn is_cargo_manifest_file_name(file_name: &str) -> bool {
    file_name == "Cargo.toml"
}

/// Whether a file name is a clippy configuration file (clippy reads either).
pub(crate) fn is_clippy_configuration_file_name(file_name: &str) -> bool {
    file_name == "clippy.toml" || file_name == ".clippy.toml"
}

/// Every way one Cargo manifest lets the function-length lint go unchecked.
pub(crate) fn manifest_exemptions(
    relative_path: &str,
    manifest_text: &str,
) -> Vec<FunctionLengthExemption> {
    let lint_settings = read_manifest_lint_settings(manifest_text);
    let mut exemptions = Vec::new();
    for line_number in &lint_settings.non_deny_lint_lines {
        exemptions.push(exemption(
            relative_path,
            *line_number,
            ExemptionKind::ManifestLintLevel,
        ));
    }
    let declares_lint_owner = lint_settings.is_package || lint_settings.is_workspace_root;
    if declares_lint_owner && !lint_settings.inherits_workspace_lints && !lint_settings.denies_lint
    {
        exemptions.push(exemption(
            relative_path,
            1,
            ExemptionKind::ManifestWithoutDeny,
        ));
    }
    exemptions
}

/// A clippy configuration file anywhere below the repository root.
pub(crate) fn nested_clippy_configuration_exemption(
    relative_path: &str,
) -> FunctionLengthExemption {
    exemption(relative_path, 1, ExemptionKind::NestedClippyConfiguration)
}

/// The root `clippy.toml` must set the threshold to the budget, so lowering or
/// raising the budget is one deliberate edit in two named places.
pub(crate) fn root_clippy_configuration_exemptions(
    root_configuration_text: Option<&str>,
) -> Vec<FunctionLengthExemption> {
    let expected_line = format!("too-many-lines-threshold = {FUNCTION_LENGTH_BUDGET_CODE_LINES}");
    let Some(configuration_text) = root_configuration_text else {
        return vec![exemption(
            "clippy.toml",
            1,
            ExemptionKind::BudgetThresholdMismatch,
        )];
    };
    for (line_index, configuration_line) in configuration_text.lines().enumerate() {
        if configuration_line
            .trim()
            .starts_with("too-many-lines-threshold")
        {
            if configuration_line.trim() == expected_line {
                return Vec::new();
            }
            return vec![exemption(
                "clippy.toml",
                line_index + 1,
                ExemptionKind::BudgetThresholdMismatch,
            )];
        }
    }
    vec![exemption(
        "clippy.toml",
        1,
        ExemptionKind::BudgetThresholdMismatch,
    )]
}

/// What one manifest says about the function-length lint.
#[derive(Default)]
struct ManifestLintSettings {
    is_package: bool,
    is_workspace_root: bool,
    inherits_workspace_lints: bool,
    denies_lint: bool,
    non_deny_lint_lines: Vec<usize>,
}

fn read_manifest_lint_settings(manifest_text: &str) -> ManifestLintSettings {
    let mut lint_settings = ManifestLintSettings::default();
    let mut current_section = String::new();
    for (line_index, manifest_line) in manifest_text.lines().enumerate() {
        let trimmed_line = manifest_line.trim();
        if let Some(section_name) = section_header_name(trimmed_line) {
            current_section = section_name.to_owned();
            lint_settings.is_package |= current_section == "package";
            lint_settings.is_workspace_root |= current_section == "workspace";
            continue;
        }
        read_lint_setting_line(
            &mut lint_settings,
            &current_section,
            trimmed_line,
            line_index + 1,
        );
    }
    lint_settings
}

fn read_lint_setting_line(
    lint_settings: &mut ManifestLintSettings,
    current_section: &str,
    trimmed_line: &str,
    line_number: usize,
) {
    let Some((setting_name, setting_value)) = trimmed_line.split_once('=') else {
        return;
    };
    let setting_name = setting_name.trim();
    if current_section == "lints" && setting_name == "workspace" {
        lint_settings.inherits_workspace_lints |= setting_value.trim() == "true";
        return;
    }
    let is_clippy_lints_section =
        current_section == "lints.clippy" || current_section == "workspace.lints.clippy";
    if !is_clippy_lints_section || setting_name != FUNCTION_LENGTH_LINT_NAME {
        return;
    }
    if setting_value.contains("\"deny\"") {
        lint_settings.denies_lint = true;
    } else {
        lint_settings.non_deny_lint_lines.push(line_number);
    }
}

/// `[lints.clippy]` gives `lints.clippy` and `[[bin]]` gives `bin`; any line
/// that is not a table header gives `None`.
fn section_header_name(trimmed_line: &str) -> Option<&str> {
    let section_name = trimmed_line.strip_prefix('[')?.strip_suffix(']')?;
    Some(section_name.trim_matches(['[', ']']).trim())
}

fn exemption(
    relative_path: &str,
    line_number: usize,
    kind: ExemptionKind,
) -> FunctionLengthExemption {
    FunctionLengthExemption {
        relative_path: relative_path.to_owned(),
        line_number,
        kind,
    }
}

#[cfg(test)]
#[path = "lint_settings_scan_tests.rs"]
mod tests;
