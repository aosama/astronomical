//! The rules `cargo xtask check-architecture` holds every family's
//! recognition crate to, read from the folder layout (issue #2473, Step 5).
//!
//! A family keeps how discovery recognizes its folders in a crate at
//! `crates/families/<kind>/<family>/recognition/`, beside its engine. The
//! daemon discovers models through those crates and never links MLX, so none
//! of them may reach the MLX bindings; and like every family crate they stay
//! off the serving crate. Naming no crate means a new family's recognition
//! crate is held to both rules the moment it exists.

use std::collections::BTreeSet;

use crate::check_architecture::{
    ArchitectureViolation, dependency_graph_from_metadata, package_manifest_paths,
    shortest_dependency_chain,
};

const RECOGNITION_MANIFEST_MARKER: &str = "/crates/families/";
const RECOGNITION_FOLDER_NAME: &str = "recognition";

pub(crate) const RECOGNITION_WITHOUT_MLX_RULE: &str = "recognition-without-mlx-bindings";
const RECOGNITION_WITHOUT_MLX_RATIONALE: &str = "a family's recognition crate must build without the MLX bindings, because the daemon discovers models through it and never links MLX";

pub(crate) const RECOGNITION_WITHOUT_SERVING_RULE: &str = "recognition-without-serving";
const RECOGNITION_WITHOUT_SERVING_RATIONALE: &str = "a family's recognition crate must build without the serving crate, which sits above every family";

/// Every rule this module checks, for the checker's pass lines.
pub(crate) const RECOGNITION_RULE_NAMES: [&str; 2] = [
    RECOGNITION_WITHOUT_MLX_RULE,
    RECOGNITION_WITHOUT_SERVING_RULE,
];

/// The crates each recognition crate must never reach, with the rule that
/// forbids it.
const FORBIDDEN_CRATES: [(&str, &str, &str); 2] = [
    (
        "mlx-bridge",
        RECOGNITION_WITHOUT_MLX_RULE,
        RECOGNITION_WITHOUT_MLX_RATIONALE,
    ),
    (
        "model-serving",
        RECOGNITION_WITHOUT_SERVING_RULE,
        RECOGNITION_WITHOUT_SERVING_RATIONALE,
    ),
];

/// Checks the recognition rules against a `cargo metadata --format-version 1`
/// document. Fails closed when no recognition crate exists or a forbidden
/// crate is absent, so a folder rename cannot make the rules match nothing.
pub(crate) fn recognition_crate_violations(
    metadata_json_text: &str,
) -> Result<Vec<ArchitectureViolation>, String> {
    let dependency_graph = dependency_graph_from_metadata(metadata_json_text)?;
    let recognition_crates = recognition_crate_names(metadata_json_text)?;
    if recognition_crates.is_empty() {
        return Err(format!(
            "[check-architecture] no crate sits at crates/families/<kind>/<family>/{RECOGNITION_FOLDER_NAME}; a folder rename must update the recognition rules"
        ));
    }
    let mut violations = Vec::new();
    for (forbidden_crate, rule_name, rationale) in FORBIDDEN_CRATES {
        if !dependency_graph.contains_key(forbidden_crate) {
            return Err(format!(
                "[check-architecture] rule={rule_name} references unknown crate {forbidden_crate}; rename drift must update the rule"
            ));
        }
        for recognition_crate in &recognition_crates {
            if let Some(dependency_chain) =
                shortest_dependency_chain(&dependency_graph, recognition_crate, forbidden_crate)
            {
                violations.push(ArchitectureViolation {
                    rule_name,
                    rationale,
                    dependency_chain,
                });
            }
        }
    }
    Ok(violations)
}

/// The packages whose manifest sits at
/// `.../crates/families/<kind>/<family>/recognition/Cargo.toml`.
fn recognition_crate_names(metadata_json_text: &str) -> Result<BTreeSet<String>, String> {
    let mut recognition_crates = BTreeSet::new();
    for (package_name, manifest_path) in package_manifest_paths(metadata_json_text)? {
        if is_recognition_manifest(&manifest_path) {
            recognition_crates.insert(package_name);
        }
    }
    Ok(recognition_crates)
}

fn is_recognition_manifest(manifest_path: &str) -> bool {
    let Some((_, path_below_families)) = manifest_path.split_once(RECOGNITION_MANIFEST_MARKER)
    else {
        return false;
    };
    let path_segments: Vec<&str> = path_below_families.split('/').collect();
    path_segments.len() == 4
        && path_segments[2] == RECOGNITION_FOLDER_NAME
        && path_segments[3] == "Cargo.toml"
}
