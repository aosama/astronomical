//! Narrows a cargo alias from `.cargo/config.toml` to the selected packages.
//!
//! The lane aliases name every tested package with `-p` and their test
//! targets with `--test`. cargo refuses a `--test` target that no selected
//! package has, so a target is kept only when a kept package defines it.
//! When no package is left, or every requested test target is gone and the
//! alias asked for nothing else, there is nothing to compile.

use std::collections::BTreeSet;

use super::workspace_graph::WorkspaceGraph;

/// Flags that select targets on their own, so the alias still builds
/// something when all of its `--test` targets were dropped.
const STANDALONE_TARGET_FLAGS: &[&str] = &["--lib", "--tests", "--bins", "--all-targets"];

/// The value of `alias_name = "..."` in the `[alias]` table, if present.
pub(crate) fn alias_text_from_config(config_text: &str, alias_name: &str) -> Option<String> {
    let mut is_in_alias_table = false;
    for config_line in config_text.lines() {
        let trimmed_line = config_line.trim();
        if trimmed_line.starts_with('[') {
            is_in_alias_table = trimmed_line == "[alias]";
            continue;
        }
        if !is_in_alias_table {
            continue;
        }
        let Some((line_name, line_value)) = trimmed_line.split_once('=') else {
            continue;
        };
        if line_name.trim() == alias_name {
            return Some(line_value.trim().trim_matches('"').to_owned());
        }
    }
    None
}

/// An alias's words. The lane aliases quote nothing, so whitespace splits them.
pub(crate) fn alias_arguments(alias_text: &str) -> Vec<String> {
    alias_text.split_whitespace().map(str::to_owned).collect()
}

/// The alias's arguments followed by `extra_arguments`, narrowed to
/// `selected_package_names`, or None when nothing would be compiled.
pub(crate) fn narrowed_alias_arguments(
    alias_text: &str,
    extra_arguments: &[String],
    selected_package_names: &BTreeSet<String>,
    workspace_graph: &WorkspaceGraph,
) -> Option<Vec<String>> {
    let mut all_arguments = alias_arguments(alias_text);
    all_arguments.extend(extra_arguments.iter().cloned());
    let kept_package_names = kept_packages(&all_arguments, selected_package_names);
    if kept_package_names.is_empty() {
        return None;
    }
    let available_test_targets = test_targets_of(&kept_package_names, workspace_graph);
    let (narrowed_arguments, requested_test_count, kept_test_count) =
        narrow_arguments(&all_arguments, &kept_package_names, &available_test_targets);
    let has_standalone_target = narrowed_arguments
        .iter()
        .any(|argument| STANDALONE_TARGET_FLAGS.contains(&argument.as_str()));
    if requested_test_count > 0 && kept_test_count == 0 && !has_standalone_target {
        return None;
    }
    Some(with_archive_profile_for_selection(
        narrowed_arguments,
        &kept_package_names,
        super::REST_CONTRACTS_PACKAGE,
    ))
}

/// The nextest profile the CI archives use. It packs the daemon's
/// executables, which the REST and end-to-end journeys launch.
const CI_ARCHIVE_PROFILE: &str = "ci";
/// The same archive without the daemon's executables, for a selection that
/// leaves the daemon out: the daemon is not built then, and no lane that
/// runs from that archive launches it.
const CI_ARCHIVE_PROFILE_WITHOUT_DAEMON_EXECUTABLES: &str = "ci-without-daemon-executables";

/// Switches a `--profile ci` archive to the profile without the daemon's
/// executables when the daemon package is not kept. Without this, the
/// archive step demanded executables the narrowed build never made.
pub(crate) fn with_archive_profile_for_selection(
    mut narrowed_arguments: Vec<String>,
    kept_package_names: &BTreeSet<String>,
    daemon_package_name: &str,
) -> Vec<String> {
    if kept_package_names.contains(daemon_package_name) {
        return narrowed_arguments;
    }
    for argument_index in 1..narrowed_arguments.len() {
        if narrowed_arguments[argument_index - 1] == "--profile"
            && narrowed_arguments[argument_index] == CI_ARCHIVE_PROFILE
        {
            narrowed_arguments[argument_index] =
                CI_ARCHIVE_PROFILE_WITHOUT_DAEMON_EXECUTABLES.to_owned();
        }
    }
    narrowed_arguments
}

fn kept_packages(
    all_arguments: &[String],
    selected_package_names: &BTreeSet<String>,
) -> BTreeSet<String> {
    let mut kept_package_names = BTreeSet::new();
    for argument_pair in all_arguments.windows(2) {
        if argument_pair[0] == "-p" && selected_package_names.contains(&argument_pair[1]) {
            kept_package_names.insert(argument_pair[1].clone());
        }
    }
    kept_package_names
}

fn test_targets_of(
    package_names: &BTreeSet<String>,
    workspace_graph: &WorkspaceGraph,
) -> BTreeSet<String> {
    let mut test_target_names = BTreeSet::new();
    for package_name in package_names {
        if let Some(workspace_package) = workspace_graph.packages.get(package_name) {
            test_target_names.extend(workspace_package.test_target_names.iter().cloned());
        }
    }
    test_target_names
}

/// Drops `-p` pairs for packages not kept and `--test` pairs for targets no
/// kept package has; returns the arguments and the requested and kept test
/// target counts.
fn narrow_arguments(
    all_arguments: &[String],
    kept_package_names: &BTreeSet<String>,
    available_test_targets: &BTreeSet<String>,
) -> (Vec<String>, usize, usize) {
    let mut narrowed_arguments = Vec::new();
    let (mut requested_test_count, mut kept_test_count) = (0, 0);
    let mut argument_index = 0;
    while argument_index < all_arguments.len() {
        let argument = &all_arguments[argument_index];
        let paired_value = all_arguments.get(argument_index + 1);
        let keeps_pair = match (argument.as_str(), paired_value) {
            ("-p", Some(package_name)) => Some(kept_package_names.contains(package_name)),
            ("--test", Some(test_target_name)) => {
                requested_test_count += 1;
                let is_available = available_test_targets.contains(test_target_name);
                kept_test_count += usize::from(is_available);
                Some(is_available)
            }
            _ => None,
        };
        match (keeps_pair, paired_value) {
            (Some(true), Some(paired_value)) => {
                narrowed_arguments.extend([argument.clone(), paired_value.clone()])
            }
            (Some(false), _) => {}
            _ => {
                narrowed_arguments.push(argument.clone());
                argument_index += 1;
                continue;
            }
        }
        argument_index += 2;
    }
    (narrowed_arguments, requested_test_count, kept_test_count)
}
