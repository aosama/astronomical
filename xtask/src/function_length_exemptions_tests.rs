use std::fs;
use std::path::Path;

use crate::function_length_exemptions::{
    ExemptionKind, FunctionLengthExemption, exemption_line_numbers, exemption_report_lines,
    find_function_length_exemptions,
};

#[test]
fn should_find_an_expect_on_a_production_function() {
    let source_text = "fn short() {}\n\n#[expect(clippy::too_many_lines)]\nfn long() {}\n";

    assert_eq!(exemption_line_numbers(source_text), [3]);
}

#[test]
fn should_find_a_module_wide_allow_and_a_cfg_attr_allow() {
    // Built with concat! so no line of this file starts with the attribute
    // itself, which the repository-wide scan would report.
    let source_text = concat!(
        "#![allow(clippy::too_many_lines)]\n",
        "#[cfg_attr(feature = \"direct-mlx\", allow(clippy::too_many_lines))]\n",
        "fn long() {}\n",
    );

    assert_eq!(exemption_line_numbers(source_text), [1, 2]);
}

#[test]
fn should_find_an_attribute_that_spans_several_lines_at_its_first_line() {
    let source_text =
        "#[expect(\n    clippy::too_many_lines,\n    reason = \"table\"\n)]\nfn long() {}\n";

    assert_eq!(exemption_line_numbers(source_text), [1]);
}

#[test]
fn should_ignore_a_deny_or_warn_of_the_lint() {
    let source_text =
        "#[deny(clippy::too_many_lines)]\nfn a() {}\n#[warn(clippy::too_many_lines)]\nfn b() {}\n";

    assert!(exemption_line_numbers(source_text).is_empty());
}

#[test]
fn should_ignore_the_lint_named_in_a_comment_or_a_string() {
    let source_text = "// #[expect(clippy::too_many_lines)] was removed\n\
                       const LINT: &str = \"#[allow(clippy::too_many_lines)]\";\n";

    assert!(exemption_line_numbers(source_text).is_empty());
}

#[test]
fn should_find_an_expect_inside_an_inline_test_module() {
    let source_text = "fn production() {}\n\n#[cfg(test)]\nmod tests {\n    #[test]\n    #[expect(clippy::too_many_lines)]\n    fn long_test() {}\n}\n";

    assert_eq!(exemption_line_numbers(source_text), [6]);
}

#[test]
fn should_keep_scanning_after_a_test_module_declared_in_another_file() {
    let source_text = "#[cfg(test)]\n#[path = \"thing_tests.rs\"]\nmod tests;\n\n#[expect(clippy::too_many_lines)]\nfn long() {}\n";

    assert_eq!(exemption_line_numbers(source_text), [5]);
}

#[test]
fn should_find_nothing_in_an_empty_source() {
    assert!(exemption_line_numbers("").is_empty());
}

#[test]
fn should_find_an_allow_of_the_pedantic_group_that_holds_the_lint() {
    let source_text = concat!("#![allow(clippy::pedantic)]\n", "fn long() {}\n");

    assert_eq!(exemption_line_numbers(source_text), [1]);
}

#[test]
fn should_close_an_attribute_whose_string_holds_an_unmatched_bracket() {
    let source_text = concat!(
        "#[error(\"range [{start}, {end})\")]\n",
        "struct RangeError;\n",
        "#[expect(clippy::too_many_lines)]\n",
        "fn long() {}\n",
    );

    assert_eq!(exemption_line_numbers(source_text), [3]);
}

#[test]
fn should_close_an_attribute_whose_trailing_comment_holds_a_bracket() {
    let source_text = concat!(
        "#[test] // see [the issue\n",
        "fn short() {}\n",
        "#[allow(clippy::too_many_lines)]\n",
        "fn long() {}\n",
    );

    assert_eq!(exemption_line_numbers(source_text), [3]);
}

#[test]
fn should_name_the_reason_location_and_remedy_in_each_report_line() {
    let exemptions = [FunctionLengthExemption {
        relative_path: "crates/engine/engine-core/src/thing.rs".to_owned(),
        line_number: 42,
        kind: ExemptionKind::LintAttribute,
    }];

    let report_lines = exemption_report_lines(&exemptions);

    assert_eq!(report_lines.len(), 1);
    assert!(report_lines[0].contains("reason=lint-exemption"));
    assert!(report_lines[0].contains("location=crates/engine/engine-core/src/thing.rs:42"));
    assert!(report_lines[0].contains("hint=split the function into named steps of at most 80"));
}

#[test]
fn should_report_production_and_test_files_alike() {
    let repository_directory = repository_with_budgeted_clippy_configuration();
    let crate_directory = repository_directory.path().join("crates/engine/thing");
    let exempted_function = "#[expect(clippy::too_many_lines)]\nfn long() {}\n";
    write_file(&crate_directory.join("src/lib.rs"), exempted_function);
    write_file(&crate_directory.join("src/short.rs"), "fn short() {}\n");
    write_file(&crate_directory.join("tests/journey.rs"), exempted_function);

    let (exemptions, _) = find_function_length_exemptions(repository_directory.path())
        .expect("the scan should read every file");

    assert_eq!(
        exemptions,
        [
            attribute_exemption("crates/engine/thing/src/lib.rs"),
            attribute_exemption("crates/engine/thing/tests/journey.rs"),
        ]
    );
}

#[test]
fn should_skip_vendored_third_party_sources() {
    let repository_directory = repository_with_budgeted_clippy_configuration();
    let exempted_function = "#[expect(clippy::too_many_lines)]\nfn long() {}\n";
    write_file(
        &repository_directory
            .path()
            .join("crates/engine/thing/third-party/pinned.rs"),
        exempted_function,
    );

    let (exemptions, scanned_file_count) =
        find_function_length_exemptions(repository_directory.path())
            .expect("the scan should read every file");

    assert!(exemptions.is_empty());
    assert_eq!(scanned_file_count, 1, "only the root clippy.toml is read");
}

#[test]
fn should_report_a_manifest_and_a_nested_clippy_configuration_that_relax_the_lint() {
    let repository_directory = repository_with_budgeted_clippy_configuration();
    let crate_directory = repository_directory.path().join("crates/engine/thing");
    write_file(
        &crate_directory.join("Cargo.toml"),
        "[package]\nname = \"thing\"\n\n[lints.clippy]\ntoo_many_lines = \"allow\"\n",
    );
    write_file(
        &crate_directory.join("clippy.toml"),
        "too-many-lines-threshold = 200\n",
    );

    let (exemptions, _) = find_function_length_exemptions(repository_directory.path())
        .expect("the scan should read every file");

    let mut exemption_kinds = Vec::new();
    for exemption in &exemptions {
        exemption_kinds.push(exemption.kind);
    }
    assert_eq!(
        exemption_kinds,
        [
            ExemptionKind::ManifestLintLevel,
            ExemptionKind::ManifestWithoutDeny,
            ExemptionKind::NestedClippyConfiguration,
        ]
    );
}

#[cfg(unix)]
#[test]
fn should_fail_naming_a_source_that_cannot_be_read() {
    use std::os::unix::fs::PermissionsExt;

    let repository_directory = repository_with_budgeted_clippy_configuration();
    let unreadable_source = repository_directory
        .path()
        .join("crates/engine/thing/src/lib.rs");
    write_file(&unreadable_source, "fn short() {}\n");
    fs::set_permissions(&unreadable_source, fs::Permissions::from_mode(0o000))
        .expect("the source should accept new permissions");

    let scan_error = find_function_length_exemptions(repository_directory.path())
        .expect_err("an unreadable source should fail the scan");

    fs::set_permissions(&unreadable_source, fs::Permissions::from_mode(0o644))
        .expect("the source should accept its permissions back so the folder can be removed");
    assert!(scan_error.contains("could not read crates/engine/thing/src/lib.rs"));
}

fn repository_with_budgeted_clippy_configuration() -> tempfile::TempDir {
    let repository_directory = tempfile::tempdir().expect("a temporary repository");
    write_file(
        &repository_directory.path().join("clippy.toml"),
        "too-many-lines-threshold = 80\n",
    );
    repository_directory
}

fn write_file(file_path: &Path, file_text: &str) {
    let parent_directory = file_path.parent().expect("every written file has a folder");
    fs::create_dir_all(parent_directory).expect("the file's folder");
    fs::write(file_path, file_text).expect("the file");
}

fn attribute_exemption(relative_path: &str) -> FunctionLengthExemption {
    FunctionLengthExemption {
        relative_path: relative_path.to_owned(),
        line_number: 1,
        kind: ExemptionKind::LintAttribute,
    }
}
