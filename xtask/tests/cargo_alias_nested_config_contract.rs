//! Every alias in `.cargo/config.toml` must expand the same whether cargo
//! finds that file once in its config ancestry or twice. Cargo merges the
//! config of every ancestor directory: it JOINS array-valued aliases across
//! files, so a worktree nested under the main checkout ran `check ... check
//! ...` (issue #1363), while a string-valued alias from the nearest file
//! replaces the outer one.
//!
//! Typed port of the former
//! `verification/tooling_contracts/test-cargo-alias-nested-config-contract.sh`
//! (issue #2635). Each case copies a config into an outer and a nested inner
//! sandbox directory and compares the alias listing `cargo --list` prints
//! from each. The sandboxes live under the temporary directory, outside any
//! checkout, so the only config files in their ancestry are the planted ones.

mod contract_support;

use contract_support::repository_root;
use std::fs;
use std::path::Path;
use std::process::Command;

const CARGO_CONFIG_PATH: &str = ".cargo/config.toml";
const ALIAS_LISTING_MARKER: &str = " alias: ";

/// The `name alias: expansion` lines `cargo --list` prints from a directory;
/// cargo's ancestry walk starts at the working directory.
fn alias_listing_from(listing_directory: &Path) -> Vec<String> {
    let cargo_list_output = Command::new("cargo")
        .arg("--list")
        .current_dir(listing_directory)
        .output()
        .expect("cargo should run");
    assert!(
        cargo_list_output.status.success(),
        "cargo --list failed in {}: {}",
        listing_directory.display(),
        String::from_utf8_lossy(&cargo_list_output.stderr)
    );
    let mut alias_lines = Vec::new();
    for listing_line in String::from_utf8_lossy(&cargo_list_output.stdout).lines() {
        if listing_line.contains(ALIAS_LISTING_MARKER) {
            alias_lines.push(listing_line.to_owned());
        }
    }
    alias_lines
}

fn has_line_containing(alias_lines: &[String], expected_text: &str) -> bool {
    for alias_line in alias_lines {
        if alias_line.contains(expected_text) {
            return true;
        }
    }
    false
}

/// The listings from the outer and the nested inner directory, after planting
/// `config_text` in both. The checkout's toolchain pin goes in the outer
/// directory, so cargo in the sandbox is the same cargo as in the checkout and
/// rustup never falls back to another toolchain.
fn outer_and_inner_listings(config_text: &str) -> (Vec<String>, Vec<String>) {
    let sandbox = tempfile::tempdir().expect("a sandbox directory should be creatable");
    let outer_directory = sandbox.path().join("outer");
    let inner_directory = outer_directory.join("inner");
    for planted_directory in [&outer_directory, &inner_directory] {
        fs::create_dir_all(planted_directory.join(".cargo"))
            .expect("the planted .cargo directory should be creatable");
        fs::write(planted_directory.join(CARGO_CONFIG_PATH), config_text)
            .expect("the planted config should be writable");
    }
    fs::copy(
        repository_root().join("rust-toolchain.toml"),
        outer_directory.join("rust-toolchain.toml"),
    )
    .expect("the toolchain pin should copy into the sandbox");
    (
        alias_listing_from(&outer_directory),
        alias_listing_from(&inner_directory),
    )
}

/// Whether the listing resolves the repository's `verify` alias; cargo pads
/// the name column, so the line is read as its first word and its ending.
fn lists_the_verify_alias(alias_lines: &[String]) -> bool {
    for alias_line in alias_lines {
        if alias_line.split_whitespace().next() == Some("verify")
            && alias_line.ends_with("alias: check --workspace --all-targets")
        {
            return true;
        }
    }
    false
}

#[test]
fn should_double_an_array_alias_when_its_config_is_found_twice() {
    // The positive control: unless the comparison sees an array alias double,
    // the repository case below proves nothing.
    let (outer_listing, inner_listing) =
        outer_and_inner_listings("[alias]\nfixture-array-alias = [\"check\", \"--workspace\"]\n");
    assert_ne!(outer_listing, inner_listing);
    assert!(
        has_line_containing(&inner_listing, "check --workspace check --workspace"),
        "the nested listing did not show the doubled array alias: {inner_listing:?}"
    );
}

#[test]
fn should_keep_a_string_alias_when_its_config_is_found_twice() {
    let (outer_listing, inner_listing) =
        outer_and_inner_listings("[alias]\nfixture-string-alias = \"check --workspace\"\n");
    assert_eq!(outer_listing, inner_listing);
}

#[test]
fn should_expand_every_repository_alias_the_same_when_the_config_is_found_twice() {
    let repository_config = fs::read_to_string(repository_root().join(CARGO_CONFIG_PATH))
        .expect("the repository cargo config should be readable");
    let (outer_listing, inner_listing) = outer_and_inner_listings(&repository_config);
    // An empty listing would pass the comparison without proving anything.
    assert!(
        lists_the_verify_alias(&outer_listing),
        "the sandbox listing did not resolve the repository's verify alias"
    );
    assert_eq!(
        outer_listing, inner_listing,
        "an alias in .cargo/config.toml changes meaning when the config file appears twice in cargo's ancestry; write every alias as one string, not a TOML array"
    );
}
