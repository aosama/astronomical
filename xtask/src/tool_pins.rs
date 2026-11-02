//! Reads the verification-tool versions from `third-party/tool-pins/Cargo.toml`,
//! the one place sccache and cargo-about are pinned. The pins live in a Cargo
//! manifest rather than plain-text files so Dependabot's Cargo support opens
//! bump pull requests for them; the installer and the notices check both read
//! the exact version through this module. Parsing is pure and unit-tested
//! beside this source; only `read_tool_pin` touches the filesystem.

use std::path::Path;

pub const TOOL_PINS_MANIFEST_RELATIVE_PATH: &str = "third-party/tool-pins/Cargo.toml";
const VERSION_KEY: &str = "version";

/// A pin must carry exactly one dotted version.
pub fn read_dotted_version_pin(pin_text: &str) -> Result<String, String> {
    let pinned_version = pin_text.trim();
    if is_dotted_version_text(pinned_version) {
        Ok(pinned_version.to_owned())
    } else {
        Err("pin must be a dotted version".to_owned())
    }
}

/// Non-empty text made only of ASCII digits and dots.
fn is_dotted_version_text(version_text: &str) -> bool {
    let mut has_only_digits_and_dots = !version_text.is_empty();
    for version_byte in version_text.bytes() {
        has_only_digits_and_dots &= version_byte.is_ascii_digit() || version_byte == b'.';
    }
    has_only_digits_and_dots
}

/// The exact `=X.Y.Z` version `crate_name` is pinned to under
/// `[dependencies]`. Ranges and caret requirements are rejected: the
/// installer downloads one release, so the manifest must name exactly one.
pub fn exact_tool_pin(manifest_text: &str, crate_name: &str) -> Result<String, String> {
    let Some(dependency_value) = dependency_value_for(manifest_text, crate_name) else {
        return Err(format!("{crate_name} is not pinned under [dependencies]"));
    };
    let Some(version_requirement) = version_requirement_of(dependency_value) else {
        return Err(format!(
            "{crate_name} pin has no quoted version requirement"
        ));
    };
    let Some(exact_version) = version_requirement.strip_prefix('=') else {
        return Err(format!(
            "{crate_name} pin must be an exact =X.Y.Z requirement, got {version_requirement}"
        ));
    };
    match read_dotted_version_pin(exact_version) {
        Ok(pinned_version) => Ok(pinned_version),
        Err(pin_error) => Err(format!("{crate_name} {pin_error}")),
    }
}

/// The text after `crate_name =` on its `[dependencies]` line. A name that
/// merely shares a prefix (`sccache-dist` for `sccache`) does not match.
fn dependency_value_for<'manifest>(
    manifest_text: &'manifest str,
    crate_name: &str,
) -> Option<&'manifest str> {
    let mut is_inside_dependencies = false;
    for manifest_line in manifest_text.lines() {
        let trimmed_line = manifest_line.trim();
        if trimmed_line.starts_with('[') {
            is_inside_dependencies = trimmed_line == "[dependencies]";
            continue;
        }
        if !is_inside_dependencies {
            continue;
        }
        let Some(after_crate_name) = trimmed_line.strip_prefix(crate_name) else {
            continue;
        };
        if let Some(dependency_value) = after_crate_name.trim_start().strip_prefix('=') {
            return Some(dependency_value.trim_start());
        }
    }
    None
}

/// The quoted requirement from either `"=1.2.3"` or
/// `{ version = "=1.2.3", ... }`.
fn version_requirement_of(dependency_value: &str) -> Option<&str> {
    if dependency_value.starts_with('"') {
        first_quoted_text(dependency_value)
    } else {
        first_quoted_text(table_version_value(dependency_value)?)
    }
}

/// The text after `version =` inside an inline dependency table.
fn table_version_value(inline_table_text: &str) -> Option<&str> {
    let version_key_start = inline_table_text.find(VERSION_KEY)?;
    let after_version_key = &inline_table_text[version_key_start + VERSION_KEY.len()..];
    Some(
        after_version_key
            .trim_start()
            .strip_prefix('=')?
            .trim_start(),
    )
}

/// The contents of the string literal `value_text` starts with.
fn first_quoted_text(value_text: &str) -> Option<&str> {
    let after_opening_quote = value_text.strip_prefix('"')?;
    let closing_quote_index = after_opening_quote.find('"')?;
    Some(&after_opening_quote[..closing_quote_index])
}

/// Reads the pin for `crate_name` from the repository's tool-pin manifest.
pub fn read_tool_pin(repository_root: &Path, crate_name: &str) -> Result<String, String> {
    let manifest_path = repository_root.join(TOOL_PINS_MANIFEST_RELATIVE_PATH);
    match std::fs::read_to_string(&manifest_path) {
        Ok(manifest_text) => exact_tool_pin(&manifest_text, crate_name),
        Err(read_error) => Err(format!(
            "missing tool-pin manifest {}: {read_error}",
            manifest_path.display()
        )),
    }
}
