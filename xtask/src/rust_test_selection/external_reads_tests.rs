use std::collections::BTreeSet;

use super::external_reads::{
    climbing_string_literals, external_paths_read_by, resolve_repository_path,
};

fn known_files() -> BTreeSet<String> {
    [
        "registry/catalog.json",
        "fixtures/romeo.txt",
        "apps/menu/Tests/Fixtures/status.json",
        "crates/core/src/lib.rs",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect()
}

#[test]
fn should_find_an_include_str_path_relative_to_the_package() {
    let source_text = r#"const CATALOG: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../registry/catalog.json"));"#;
    assert_eq!(
        external_paths_read_by(
            "crates/core",
            "crates/core/src/lib.rs",
            source_text,
            &known_files()
        ),
        BTreeSet::from(["registry/catalog.json".to_owned()])
    );
}

#[test]
fn should_find_a_path_relative_to_the_source_file_and_a_folder_read() {
    let source_text = r#"let fixture = include_str!("../../../apps/menu/Tests/Fixtures/status.json"); let folder = "../../../fixtures";"#;
    assert_eq!(
        external_paths_read_by(
            "crates/core",
            "crates/core/src/lib.rs",
            source_text,
            &known_files()
        ),
        BTreeSet::from([
            "apps/menu/Tests/Fixtures/status.json".to_owned(),
            "fixtures".to_owned()
        ])
    );
}

#[test]
fn should_ignore_paths_inside_the_package_above_the_repository_or_unknown() {
    let source_text = r#"let own = "../src/lib.rs"; let escape = "../../../../outside"; let missing = "../../registry/gone.json"; let root = "../..";"#;
    assert_eq!(
        external_paths_read_by(
            "crates/core",
            "crates/core/src/lib.rs",
            source_text,
            &known_files()
        ),
        BTreeSet::new()
    );
}

#[test]
fn should_keep_an_escaped_quote_inside_a_literal() {
    assert_eq!(
        climbing_string_literals(r#"let text = "say \"hi\" from ../up"; let plain = "no climb";"#),
        ["say \"hi\" from ../up"]
    );
}

#[test]
fn should_resolve_dots_and_refuse_to_climb_above_the_root() {
    assert_eq!(
        resolve_repository_path("crates/core", "../../registry/./catalog.json").as_deref(),
        Some("registry/catalog.json")
    );
    assert_eq!(resolve_repository_path("crates", "../../outside"), None);
    assert_eq!(
        resolve_repository_path("", "registry").as_deref(),
        Some("registry")
    );
}
