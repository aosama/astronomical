use std::path::Path;

use crate::native_store_paths::{
    native_store_entry_directory, native_store_schema_version, parse_schema_version,
};

#[test]
fn should_place_an_identity_entry_under_the_schema_generation_of_the_default_store() {
    assert_eq!(
        native_store_entry_directory(Path::new("/home-fixture"), 3, "feedface"),
        Path::new("/home-fixture/Library/Caches/Astronomical/native-builds/v3/entries/feedface")
    );
}

#[test]
fn should_read_the_committed_schema_version_from_the_repository() {
    let repository_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let committed_schema_version = native_store_schema_version(&repository_root).unwrap();
    assert!(committed_schema_version >= 1);
}

#[test]
fn should_accept_a_schema_version_with_surrounding_whitespace() {
    assert_eq!(parse_schema_version(" 7\n"), Ok(7));
}

#[test]
fn should_refuse_a_schema_version_that_is_not_an_unsigned_integer() {
    for malformed_schema_version in ["", "v3", "-1", "3.5"] {
        assert_eq!(
            parse_schema_version(malformed_schema_version),
            Err("native build store schema version must be an unsigned integer".to_owned())
        );
    }
}

#[test]
fn should_report_a_missing_schema_version_file() {
    let schema_version_failure =
        native_store_schema_version(Path::new("/nonexistent-repository-fixture")).unwrap_err();
    assert!(schema_version_failure.starts_with("native build store schema version is unavailable"));
}
