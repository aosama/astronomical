//! `cargo xtask release-speed-table [--previous VERSION|FILE --current VERSION|FILE]`
//! prints the "Speed" section a release's notes and release issue open with,
//! comparing two release speed ledgers under `registry/release_speed/`.
//! Without arguments it compares the two newest ledgers.
//!
//! A version names that release's ledger; a path ending in `.json` names any
//! ledger file, so two builds measured in one sitting can be compared too.
//! The section is printed, never written: the release procedure pastes it
//! where it belongs, and nothing in it is typed by hand.
//!
//! The catalog does not publish these figures.

#[path = "release_speed/speed_comparison.rs"]
mod speed_comparison;
#[path = "release_speed/speed_ledger.rs"]
pub(crate) mod speed_ledger;

use std::path::{Path, PathBuf};

const USAGE: &str =
    "usage: cargo xtask release-speed-table [--previous VERSION|FILE --current VERSION|FILE]";

#[derive(Debug, Eq, PartialEq)]
enum ReleaseSpeedCommand {
    /// Both references, or neither for the two newest ledgers.
    Table {
        ledger_references: Option<(String, String)>,
    },
}

pub fn run(repository_root: &Path, arguments: &[String]) -> Result<(), String> {
    match parse_command(arguments)? {
        ReleaseSpeedCommand::Table { ledger_references } => {
            let (previous_reference, current_reference) = match ledger_references {
                Some(ledger_references) => ledger_references,
                None => two_newest_versions(repository_root)?,
            };
            let previous_ledger =
                speed_ledger::read_ledger(&ledger_file(repository_root, &previous_reference))?;
            let current_ledger =
                speed_ledger::read_ledger(&ledger_file(repository_root, &current_reference))?;
            print!(
                "{}",
                speed_comparison::release_speed_section(&previous_ledger, &current_ledger)
            );
            Ok(())
        }
    }
}

fn parse_command(arguments: &[String]) -> Result<ReleaseSpeedCommand, String> {
    let mut previous_reference = None;
    let mut current_reference = None;
    let mut remaining_arguments = arguments.iter();
    while let Some(argument) = remaining_arguments.next() {
        let reference_slot = match argument.as_str() {
            "--previous" => &mut previous_reference,
            "--current" => &mut current_reference,
            _ => return Err(format!("unknown argument {argument}; {USAGE}")),
        };
        *reference_slot = Some(
            remaining_arguments
                .next()
                .cloned()
                .ok_or_else(|| format!("{argument} needs a value; {USAGE}"))?,
        );
    }
    match (previous_reference, current_reference) {
        (Some(previous), Some(current)) => Ok(ReleaseSpeedCommand::Table {
            ledger_references: Some((previous, current)),
        }),
        (None, None) => Ok(ReleaseSpeedCommand::Table {
            ledger_references: None,
        }),
        _ => Err(format!("name both ledgers or neither; {USAGE}")),
    }
}

fn two_newest_versions(repository_root: &Path) -> Result<(String, String), String> {
    match speed_ledger::ledger_versions_newest_first(repository_root).as_slice() {
        [current_version, previous_version, ..] => {
            Ok((previous_version.clone(), current_version.clone()))
        }
        _ => Err(
            "two release speed ledgers are needed to compare; measure a release first".to_owned(),
        ),
    }
}

fn ledger_file(repository_root: &Path, ledger_reference: &str) -> PathBuf {
    if Path::new(ledger_reference)
        .extension()
        .is_some_and(|extension| extension == "json")
    {
        PathBuf::from(ledger_reference)
    } else {
        speed_ledger::ledger_path(repository_root, ledger_reference)
    }
}

#[cfg(test)]
#[path = "release_speed/release_speed_tests.rs"]
mod release_speed_tests;
