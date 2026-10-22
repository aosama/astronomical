//! `cargo xtask check-architecture` — proves crate-level layering rules over
//! the real Cargo dependency graph, replacing the memory-policy layering
//! shell contract. The reachability walk is transitive across every declared
//! dependency kind (normal, build, and dev): a forbidden crate reached
//! through any chain fails the check, which the manifest grep this replaces
//! could never see. The verdict logic is pure and unit-tested beside this
//! source; the `cargo metadata` edge stays thin (issue #1250 discipline).

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::path::Path;
use std::process::Command;

/// One forbidden-dependency rule: the dependency graph must hold no path
/// from `source_crate` to `forbidden_crate`, following every dependency
/// kind. Rules name workspace members, so a crate rename must update the
/// rule rather than let it pass vacuously — both endpoints failing closed
/// when absent enforces that.
struct ForbiddenDependencyRule {
    rule_name: &'static str,
    source_crate: &'static str,
    forbidden_crate: &'static str,
    rationale: &'static str,
}

/// The memory policy crate must stay free of the MLX binding crate: pulling
/// the graphics-processor bindings in would force the policy crate to
/// rebuild the graphics-processor world and drag the serving crate's
/// graphics-processor tests into the offline lane, defeating the extraction
/// the crate exists for.
const FORBIDDEN_DEPENDENCY_RULES: &[ForbiddenDependencyRule] = &[ForbiddenDependencyRule {
    rule_name: "policy-without-mlx-bindings",
    source_crate: "astronomical-memory-policy",
    forbidden_crate: "astronomical-mlx-c-rust",
    rationale: "the memory policy crate must build and test in the offline lane without ever pulling in the serving crate or its graphics-processor tests",
}];

/// One detected violation with the offending dependency chain, source crate
/// first. The chain names every hop so a failure is attributable from the
/// message alone.
#[derive(Debug)]
pub(crate) struct ArchitectureViolation {
    rule_name: &'static str,
    rationale: &'static str,
    pub(crate) dependency_chain: Vec<String>,
}

pub fn run(repository_root: &Path) -> Result<(), String> {
    println!("[check-architecture] phase=repository-graph status=start");
    let metadata_output = Command::new("cargo")
        .arg("metadata")
        .arg("--format-version")
        .arg("1")
        .arg("--manifest-path")
        .arg(repository_root.join("Cargo.toml"))
        .output()
        .map_err(|error| format!("cargo metadata failed to start: {error}"))?;
    if !metadata_output.status.success() {
        return Err(format!(
            "cargo metadata exited with {}: {}",
            metadata_output.status,
            String::from_utf8_lossy(&metadata_output.stderr).trim()
        ));
    }
    let metadata_json_text = String::from_utf8_lossy(&metadata_output.stdout).into_owned();
    let violations = architecture_violations(&metadata_json_text)?;
    if violations.is_empty() {
        for rule in FORBIDDEN_DEPENDENCY_RULES {
            println!("[check-architecture] rule={} status=pass", rule.rule_name);
        }
        println!(
            "[check-architecture] status=success rules={}",
            FORBIDDEN_DEPENDENCY_RULES.len()
        );
        Ok(())
    } else {
        for violation in &violations {
            eprintln!(
                "[check-architecture] rule={} violation: {}; {}",
                violation.rule_name,
                violation.dependency_chain.join(" -> "),
                violation.rationale
            );
        }
        Err(format!(
            "[check-architecture] status=failed violations={}",
            violations.len()
        ))
    }
}

/// Checks every rule against a `cargo metadata --format-version 1` document
/// and returns the violations. Fails closed when the document is malformed,
/// when cargo emits an unknown dependency kind (its format changed under
/// us), or when a rule endpoint is absent from the graph.
pub(crate) fn architecture_violations(
    metadata_json_text: &str,
) -> Result<Vec<ArchitectureViolation>, String> {
    let dependency_graph = dependency_graph_from_metadata(metadata_json_text)?;
    let mut violations = Vec::new();
    for rule in FORBIDDEN_DEPENDENCY_RULES {
        if !dependency_graph.contains_key(rule.source_crate) {
            return Err(format!(
                "[check-architecture] rule={} references unknown source crate {}; rename drift must update the rule",
                rule.rule_name, rule.source_crate
            ));
        }
        if !dependency_graph.contains_key(rule.forbidden_crate) {
            return Err(format!(
                "[check-architecture] rule={} references unknown forbidden crate {}; rename drift must update the rule",
                rule.rule_name, rule.forbidden_crate
            ));
        }
        if let Some(dependency_chain) =
            shortest_dependency_chain(&dependency_graph, rule.source_crate, rule.forbidden_crate)
        {
            violations.push(ArchitectureViolation {
                rule_name: rule.rule_name,
                rationale: rule.rationale,
                dependency_chain,
            });
        }
    }
    Ok(violations)
}

/// Extracts the crate-name adjacency from the metadata document. Dev and
/// build dependencies count as edges deliberately: a dev-edge would pull
/// the forbidden crate into the source crate's own test builds, which is
/// exactly the layering breach the rules exist to prevent.
fn dependency_graph_from_metadata(
    metadata_json_text: &str,
) -> Result<BTreeMap<String, BTreeSet<String>>, String> {
    let metadata_document: serde_json::Value = serde_json::from_str(metadata_json_text)
        .map_err(|error| format!("cargo metadata is not valid JSON: {error}"))?;
    let packages_array = metadata_document
        .get("packages")
        .and_then(|packages_value| packages_value.as_array())
        .ok_or("cargo metadata is missing the packages array")?;
    let mut dependency_graph: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for package_value in packages_array {
        let package_name = package_value
            .get("name")
            .and_then(|name_value| name_value.as_str())
            .ok_or("cargo metadata contains a package without a name")?;
        let dependency_names = dependency_graph.entry(package_name.to_owned()).or_default();
        let dependencies_array = package_value
            .get("dependencies")
            .and_then(|dependencies_value| dependencies_value.as_array())
            .ok_or_else(|| {
                format!("cargo metadata package {package_name} has no dependencies array")
            })?;
        for dependency_value in dependencies_array {
            let dependency_name = dependency_value
                .get("name")
                .and_then(|name_value| name_value.as_str())
                .ok_or_else(|| {
                    format!("cargo metadata package {package_name} has a dependency without a name")
                })?;
            let dependency_kind = dependency_value
                .get("kind")
                .map(|kind_value| kind_value.as_str().unwrap_or("normal"))
                .unwrap_or("normal");
            match dependency_kind {
                "normal" | "build" | "dev" => {
                    dependency_names.insert(dependency_name.to_owned());
                }
                unexpected_kind => {
                    return Err(format!(
                        "cargo metadata emitted an unsupported dependency kind {unexpected_kind} for {package_name} -> {dependency_name}"
                    ));
                }
            }
        }
    }
    Ok(dependency_graph)
}

/// Breadth-first search returning the shortest chain from `start_crate` to
/// `target_crate`, inclusive of both endpoints, or `None` when the target is
/// unreachable.
fn shortest_dependency_chain(
    dependency_graph: &BTreeMap<String, BTreeSet<String>>,
    start_crate: &str,
    target_crate: &str,
) -> Option<Vec<String>> {
    let mut discovered_from: BTreeMap<&str, &str> = BTreeMap::new();
    let mut visit_queue: VecDeque<&str> = VecDeque::from([start_crate]);
    discovered_from.insert(start_crate, start_crate);
    while let Some(current_crate) = visit_queue.pop_front() {
        if current_crate == target_crate {
            let mut chain_crates: Vec<&str> = Vec::new();
            let mut walk_crate = target_crate;
            while walk_crate != start_crate {
                chain_crates.push(walk_crate);
                walk_crate = discovered_from[walk_crate];
            }
            chain_crates.push(start_crate);
            chain_crates.reverse();
            return Some(chain_crates.into_iter().map(str::to_owned).collect());
        }
        for neighbor_crate in dependency_graph.get(current_crate).into_iter().flatten() {
            if !discovered_from.contains_key(neighbor_crate.as_str()) {
                discovered_from.insert(neighbor_crate, current_crate);
                visit_queue.push_back(neighbor_crate);
            }
        }
    }
    None
}
