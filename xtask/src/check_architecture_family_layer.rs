//! The model-family rules of `cargo xtask check-architecture`, read from the
//! folder layout rather than from a list of crate names (issue #2473).
//!
//! Every crate under `crates/families/<kind>/` is a family-layer crate. A
//! kind folder holds one shell crate, which carries what the kind's families
//! share, and one crate per family beside it. Which is which is read from
//! the graph: a crate that depends on another crate of its own kind folder
//! is a family crate, and one that depends on none is a shell. Naming no
//! crate is the point: a new family crate is held to these rules the moment
//! it exists, without anyone editing this checker.
//!
//! Two rules follow:
//! - no family-layer crate reaches the serving crate, which drives the
//!   families through a one-way edge;
//! - no family crate reaches another family crate of its kind, because what
//!   two families share belongs in the shell, and a family that reaches
//!   another would make the two change together.

use std::collections::{BTreeMap, BTreeSet};

use crate::check_architecture::{
    ArchitectureViolation, dependency_graph_from_metadata, package_manifest_paths,
    shortest_dependency_chain,
};

const FAMILY_FOLDER_MARKER: &str = "/crates/families/";
const SERVING_CRATE: &str = "model-serving";

pub(crate) const FAMILY_WITHOUT_SERVING_RULE: &str = "family-without-serving";
const FAMILY_WITHOUT_SERVING_RATIONALE: &str = "a model family crate must build and test without pulling in the serving crate, which drives the families through a one-way edge";

pub(crate) const FAMILY_WITHOUT_OTHER_FAMILY_RULE: &str = "family-without-other-family";
const FAMILY_WITHOUT_OTHER_FAMILY_RATIONALE: &str = "one model family must not depend on another; what they share belongs in the kind's shell crate";

/// Every rule this module checks, for the checker's pass lines.
pub(crate) const FAMILY_LAYER_RULE_NAMES: [&str; 2] = [
    FAMILY_WITHOUT_SERVING_RULE,
    FAMILY_WITHOUT_OTHER_FAMILY_RULE,
];

/// Checks the family rules against a `cargo metadata --format-version 1`
/// document. Fails closed when no crate sits under `crates/families/`, or
/// when the serving crate is absent, so a folder rename cannot make the
/// rules pass by matching nothing.
pub(crate) fn family_layer_violations(
    metadata_json_text: &str,
) -> Result<Vec<ArchitectureViolation>, String> {
    let dependency_graph = dependency_graph_from_metadata(metadata_json_text)?;
    let kind_to_member_crates = family_layer_crates_by_kind(metadata_json_text)?;
    if kind_to_member_crates.is_empty() {
        return Err(format!(
            "[check-architecture] no crate sits under {FAMILY_FOLDER_MARKER}; a folder rename must update the family rules"
        ));
    }
    if !dependency_graph.contains_key(SERVING_CRATE) {
        return Err(format!(
            "[check-architecture] rule={FAMILY_WITHOUT_SERVING_RULE} references unknown crate {SERVING_CRATE}; rename drift must update the rule"
        ));
    }
    let mut violations = Vec::new();
    for member_crates in kind_to_member_crates.values() {
        violations.extend(serving_violations(&dependency_graph, member_crates));
        violations.extend(other_family_violations(&dependency_graph, member_crates));
    }
    Ok(violations)
}

/// Groups the family-layer crates by their kind folder, read from each
/// package's manifest path.
fn family_layer_crates_by_kind(
    metadata_json_text: &str,
) -> Result<BTreeMap<String, BTreeSet<String>>, String> {
    let mut kind_to_member_crates: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for (package_name, manifest_path) in package_manifest_paths(metadata_json_text)? {
        if let Some(kind_folder) = family_kind_folder(&manifest_path) {
            kind_to_member_crates
                .entry(kind_folder.to_owned())
                .or_default()
                .insert(package_name);
        }
    }
    Ok(kind_to_member_crates)
}

/// The kind folder of a manifest at `.../crates/families/<kind>/<crate>/Cargo.toml`,
/// or `None` for a manifest anywhere else.
fn family_kind_folder(manifest_path: &str) -> Option<&str> {
    let (_, path_below_families) = manifest_path.split_once(FAMILY_FOLDER_MARKER)?;
    let mut path_segments = path_below_families.split('/');
    let kind_folder = path_segments.next()?;
    let _crate_folder = path_segments.next()?;
    let manifest_file_name = path_segments.next()?;
    let is_crate_manifest = manifest_file_name == "Cargo.toml" && path_segments.next().is_none();
    is_crate_manifest.then_some(kind_folder)
}

fn serving_violations(
    dependency_graph: &BTreeMap<String, BTreeSet<String>>,
    member_crates: &BTreeSet<String>,
) -> Vec<ArchitectureViolation> {
    let mut violations = Vec::new();
    for member_crate in member_crates {
        if let Some(dependency_chain) =
            shortest_dependency_chain(dependency_graph, member_crate, SERVING_CRATE)
        {
            violations.push(ArchitectureViolation {
                rule_name: FAMILY_WITHOUT_SERVING_RULE,
                rationale: FAMILY_WITHOUT_SERVING_RATIONALE,
                dependency_chain,
            });
        }
    }
    violations
}

/// A family crate is one that depends directly on another crate of its kind
/// folder (its shell); a shell depends on none.
fn family_crates_of_kind<'members>(
    dependency_graph: &BTreeMap<String, BTreeSet<String>>,
    member_crates: &'members BTreeSet<String>,
) -> Vec<&'members String> {
    let mut family_crates = Vec::new();
    for member_crate in member_crates {
        let Some(direct_dependencies) = dependency_graph.get(member_crate.as_str()) else {
            continue;
        };
        if !direct_dependencies.is_disjoint(member_crates) {
            family_crates.push(member_crate);
        }
    }
    family_crates
}

fn other_family_violations(
    dependency_graph: &BTreeMap<String, BTreeSet<String>>,
    member_crates: &BTreeSet<String>,
) -> Vec<ArchitectureViolation> {
    let family_crates = family_crates_of_kind(dependency_graph, member_crates);
    let mut violations = Vec::new();
    for source_family in &family_crates {
        for other_family in &family_crates {
            if source_family == other_family {
                continue;
            }
            if let Some(dependency_chain) =
                shortest_dependency_chain(dependency_graph, source_family, other_family)
            {
                violations.push(ArchitectureViolation {
                    rule_name: FAMILY_WITHOUT_OTHER_FAMILY_RULE,
                    rationale: FAMILY_WITHOUT_OTHER_FAMILY_RATIONALE,
                    dependency_chain,
                });
            }
        }
    }
    violations
}
