use std::path::Path;

use crate::tool_pins::{exact_tool_pin, read_dotted_version_pin, read_tool_pin};

const PIN_MANIFEST: &str = r#"
# comment mentioning sccache = "=9.9.9" outside any section
[package]
name = "astronomical-tool-pins"
version = "0.0.0"

[dev-dependencies]
cargo-about = "=7.7.7"

[dependencies]
cargo-about = { version = "=0.9.3", default-features = false }
sccache-dist = "=5.5.5"
sccache = "=0.19.0"
"#;

#[test]
fn should_read_exact_pins_in_both_dependency_forms() {
    assert_eq!(
        exact_tool_pin(PIN_MANIFEST, "cargo-about"),
        Ok("0.9.3".to_owned())
    );
    assert_eq!(
        exact_tool_pin(PIN_MANIFEST, "sccache"),
        Ok("0.19.0".to_owned())
    );
}

#[test]
fn should_ignore_other_sections_and_prefix_sharing_crates() {
    let dev_only_manifest = "[dev-dependencies]\nsccache = \"=1.0.0\"\n";
    assert!(exact_tool_pin(dev_only_manifest, "sccache").is_err());
    let prefix_only_manifest = "[dependencies]\nsccache-dist = \"=1.0.0\"\n";
    assert!(exact_tool_pin(prefix_only_manifest, "sccache").is_err());
}

#[test]
fn should_reject_pins_that_are_not_one_exact_dotted_version() {
    for rejected_requirement in ["0.19.0", "^0.19.0", "=0.19.0-beta", "=", ">=0.19.0"] {
        let manifest_text = format!("[dependencies]\nsccache = \"{rejected_requirement}\"\n");
        assert!(
            exact_tool_pin(&manifest_text, "sccache").is_err(),
            "accepted {rejected_requirement}"
        );
    }
}

#[test]
fn should_reject_missing_or_unquoted_requirements() {
    assert!(exact_tool_pin("", "sccache").is_err());
    assert!(exact_tool_pin("[dependencies]\n", "sccache").is_err());
    assert!(exact_tool_pin("[dependencies]\nsccache = { path = \"x\" }\n", "sccache").is_err());
    assert!(exact_tool_pin("[dependencies]\nsccache = 1\n", "sccache").is_err());
}

#[test]
fn should_accept_only_dotted_version_text() {
    assert_eq!(read_dotted_version_pin("0.9.2\n"), Ok("0.9.2".to_owned()));
    assert_eq!(read_dotted_version_pin("1.2"), Ok("1.2".to_owned()));
    assert!(read_dotted_version_pin("").is_err());
    assert!(read_dotted_version_pin("latest\n").is_err());
    assert!(read_dotted_version_pin("0.9.2-beta").is_err());
}

#[test]
fn should_read_both_tool_pins_from_the_committed_manifest() {
    let repository_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask lives one level below the repository root");
    for crate_name in ["cargo-about", "sccache"] {
        let pinned_version = read_tool_pin(repository_root, crate_name);
        assert!(
            pinned_version.is_ok(),
            "committed manifest lost its {crate_name} pin: {pinned_version:?}"
        );
    }
}

#[test]
fn should_report_a_missing_manifest() {
    let empty_repository_root = tempfile::tempdir().expect("temporary directory");
    let missing_manifest_error = read_tool_pin(empty_repository_root.path(), "sccache");
    assert!(
        matches!(&missing_manifest_error, Err(error_text) if error_text.contains("missing tool-pin manifest")),
        "unexpected result: {missing_manifest_error:?}"
    );
}
