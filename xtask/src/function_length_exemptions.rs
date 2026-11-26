//! Finds anything that would let a function past the function-length budget.
//!
//! The clippy pass in `cargo xtask function-length-budget` fails any function
//! over the budget, but the lint can be switched off in three places: a Rust
//! attribute (`attribute_scan`), a Cargo manifest's lints table, or a
//! `clippy.toml` (`lint_settings_scan`). AGENTS.md holds production and test
//! code alike to the budget with no exemption, so this scan fails on any of
//! them in the files the workspace authors (see `workspace_source_walk`). It
//! reads text without compiling, so it runs before the clippy pass and takes
//! well under a second.

mod attribute_scan;
mod lint_settings_scan;

use std::fs;
use std::path::Path;

pub(crate) use attribute_scan::exemption_line_numbers;
use lint_settings_scan::{
    is_cargo_manifest_file_name, is_clippy_configuration_file_name, manifest_exemptions,
    nested_clippy_configuration_exemption, root_clippy_configuration_exemptions,
};

use crate::workspace_source_walk::{collect_authored_file_paths, is_rust_source_file_name};

/// The longest function AGENTS.md allows, in clippy code lines (blank and
/// comment lines are not counted). The root `clippy.toml` must set
/// `too-many-lines-threshold` to this value; the scan checks that it does.
pub(crate) const FUNCTION_LENGTH_BUDGET_CODE_LINES: usize = 80;

/// How a file exempts code from the budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExemptionKind {
    /// An `allow` or `expect` attribute naming the lint or its group.
    LintAttribute,
    /// A manifest sets the lint to a level other than `deny`.
    ManifestLintLevel,
    /// A manifest has its own lints table that leaves the lint at its
    /// allow-by-default level.
    ManifestWithoutDeny,
    /// A `clippy.toml` below the root, which clippy would read first for the
    /// crates under it.
    NestedClippyConfiguration,
    /// The root `clippy.toml` threshold is missing or differs from the budget.
    BudgetThresholdMismatch,
}

impl ExemptionKind {
    fn reason_label(self) -> &'static str {
        match self {
            Self::LintAttribute => "lint-exemption",
            Self::ManifestLintLevel => "manifest-lint-level",
            Self::ManifestWithoutDeny => "manifest-without-deny",
            Self::NestedClippyConfiguration => "nested-clippy-configuration",
            Self::BudgetThresholdMismatch => "budget-threshold-mismatch",
        }
    }

    fn remedy(self) -> String {
        match self {
            Self::LintAttribute => format!(
                "split the function into named steps of at most {FUNCTION_LENGTH_BUDGET_CODE_LINES} code lines and remove the attribute"
            ),
            Self::ManifestLintLevel | Self::ManifestWithoutDeny => {
                "inherit the workspace lints ([lints] workspace = true) or set too_many_lines = \"deny\" under [lints.clippy]".to_owned()
            }
            Self::NestedClippyConfiguration => {
                "remove it; the root clippy.toml is the only clippy configuration".to_owned()
            }
            Self::BudgetThresholdMismatch => format!(
                "set too-many-lines-threshold = {FUNCTION_LENGTH_BUDGET_CODE_LINES} in the root clippy.toml"
            ),
        }
    }
}

/// One place that exempts code from the budget.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct FunctionLengthExemption {
    pub(crate) relative_path: String,
    pub(crate) line_number: usize,
    pub(crate) kind: ExemptionKind,
}

/// Every exemption in the repository, the Rust attributes first and then the
/// lint settings, each in path and line order, with how many files were read.
pub(crate) fn find_function_length_exemptions(
    repository_root: &Path,
) -> Result<(Vec<FunctionLengthExemption>, usize), String> {
    let mut exemptions = Vec::new();
    let rust_file_count = scan_rust_sources(repository_root, &mut exemptions)?;
    let settings_file_count = scan_lint_settings(repository_root, &mut exemptions)?;
    Ok((exemptions, rust_file_count + settings_file_count))
}

fn scan_rust_sources(
    repository_root: &Path,
    exemptions: &mut Vec<FunctionLengthExemption>,
) -> Result<usize, String> {
    let source_paths = collect_authored_file_paths(repository_root, is_rust_source_file_name);
    for relative_path in &source_paths {
        let source_text = read_text(repository_root, relative_path)?;
        for line_number in exemption_line_numbers(&source_text) {
            exemptions.push(FunctionLengthExemption {
                relative_path: relative_path.clone(),
                line_number,
                kind: ExemptionKind::LintAttribute,
            });
        }
    }
    Ok(source_paths.len())
}

/// The root manifest and clippy configuration sit outside the authored roots,
/// so they are read by name; an absent root manifest leaves nothing to check.
fn scan_lint_settings(
    repository_root: &Path,
    exemptions: &mut Vec<FunctionLengthExemption>,
) -> Result<usize, String> {
    let mut manifest_paths =
        collect_authored_file_paths(repository_root, is_cargo_manifest_file_name);
    if repository_root.join("Cargo.toml").is_file() {
        manifest_paths.insert(0, "Cargo.toml".to_owned());
    }
    for relative_path in &manifest_paths {
        let manifest_text = read_text(repository_root, relative_path)?;
        exemptions.extend(manifest_exemptions(relative_path, &manifest_text));
    }
    let nested_configuration_paths =
        collect_authored_file_paths(repository_root, is_clippy_configuration_file_name);
    for relative_path in &nested_configuration_paths {
        exemptions.push(nested_clippy_configuration_exemption(relative_path));
    }
    let root_configuration_text = fs::read_to_string(repository_root.join("clippy.toml")).ok();
    exemptions.extend(root_clippy_configuration_exemptions(
        root_configuration_text.as_deref(),
    ));
    Ok(manifest_paths.len() + nested_configuration_paths.len() + 1)
}

/// Bytes that are not UTF-8 are replaced rather than refused, so one odd
/// fixture cannot stop the scan; a file that cannot be read fails it.
fn read_text(repository_root: &Path, relative_path: &str) -> Result<String, String> {
    match fs::read(repository_root.join(relative_path)) {
        Ok(file_bytes) => Ok(String::from_utf8_lossy(&file_bytes).into_owned()),
        Err(read_error) => Err(format!("could not read {relative_path}: {read_error}")),
    }
}

/// The failure lines printed for each exemption, with its remedy.
pub(crate) fn exemption_report_lines(exemptions: &[FunctionLengthExemption]) -> Vec<String> {
    let mut report_lines = Vec::with_capacity(exemptions.len());
    for exemption in exemptions {
        report_lines.push(format!(
            "function-length-budget status=failed reason={} location={}:{} hint={}",
            exemption.kind.reason_label(),
            exemption.relative_path,
            exemption.line_number,
            exemption.kind.remedy()
        ));
    }
    report_lines
}
