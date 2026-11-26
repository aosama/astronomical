use std::path::{Path, PathBuf};

use super::{ReleaseSpeedCommand, ledger_file, parse_command, two_newest_versions};
use crate::release_speed::speed_ledger::speed_ledger_tests::ledger;
use crate::release_speed::speed_ledger::write_ledger;

fn arguments(argument_texts: &[&str]) -> Vec<String> {
    argument_texts
        .iter()
        .map(|argument_text| (*argument_text).to_owned())
        .collect()
}

#[test]
fn should_read_the_previous_and_current_ledgers_in_either_order() {
    assert_eq!(
        parse_command(&arguments(&[
            "--current",
            "0.2.135",
            "--previous",
            "0.2.134"
        ])),
        Ok(ReleaseSpeedCommand::Table {
            ledger_references: Some(("0.2.134".to_owned(), "0.2.135".to_owned()))
        })
    );
}

#[test]
fn should_refuse_a_missing_ledger_a_missing_value_or_an_unknown_argument() {
    for refused_arguments in [
        vec!["--previous", "0.2.134"],
        vec!["--previous", "0.2.134", "--current"],
        vec!["--previous", "0.2.134", "--current", "0.2.135", "--all"],
        vec!["--check-catalog", "--previous", "0.2.134"],
    ] {
        assert!(
            parse_command(&arguments(&refused_arguments)).is_err(),
            "should refuse {refused_arguments:?}"
        );
    }
}

#[test]
fn should_take_a_version_as_its_ledger_and_a_json_path_as_itself() {
    let repository_root = Path::new("/repository");

    assert_eq!(
        ledger_file(repository_root, "0.2.135"),
        PathBuf::from("/repository/registry/release_speed/0.2.135.json")
    );
    assert_eq!(
        ledger_file(repository_root, "sitting/main-build.json"),
        PathBuf::from("sitting/main-build.json")
    );
}

#[test]
fn should_compare_the_two_newest_ledgers_when_none_is_named() {
    assert_eq!(
        parse_command(&[]),
        Ok(ReleaseSpeedCommand::Table {
            ledger_references: None
        })
    );
    assert_eq!(
        parse_command(&arguments(&["--check-catalog"])),
        Ok(ReleaseSpeedCommand::CheckCatalog)
    );
    let repository_root = tempfile::tempdir().expect("a temporary repository");
    assert!(
        two_newest_versions(repository_root.path()).is_err(),
        "no ledger yet"
    );
    for version in ["0.2.99", "0.2.140", "0.2.134"] {
        write_ledger(repository_root.path(), ledger(version, Vec::new())).expect("a ledger");
    }

    assert_eq!(
        two_newest_versions(repository_root.path()),
        Ok(("0.2.134".to_owned(), "0.2.140".to_owned()))
    );
}
