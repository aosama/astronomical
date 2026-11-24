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

use crate::check_architecture_family_layer::{FAMILY_LAYER_RULE_NAMES, family_layer_violations};

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
/// graphics-processor tests into the integration lane, defeating the extraction
/// the crate exists for.
const FORBIDDEN_DEPENDENCY_RULES: &[ForbiddenDependencyRule] = &[
    ForbiddenDependencyRule {
        rule_name: "policy-without-mlx-bindings",
        source_crate: "memory-policy",
        forbidden_crate: "mlx-bridge",
        rationale: "the memory policy crate must build and test in the integration lane without ever pulling in the serving crate or its graphics-processor tests",
    },
    ForbiddenDependencyRule {
        rule_name: "policy-without-mlx-fork-bindings",
        source_crate: "memory-policy",
        forbidden_crate: "mlx-c-fork",
        rationale: "the fork extension carries the same graphics-processor linkage as the stock bridge, so the memory policy crate must stay free of it too",
    },
    // The jangtq2 family crate (issue #1427) owns the weight contract end
    // to end; the serving integration (page loading, slot writes, dispatch)
    // lives in the serving crate and consumes the family's public types. A
    // dependency back from the family to the serving crate would make that
    // split circular and rebuild the serving crate on every family edit,
    // defeating the extraction.
    ForbiddenDependencyRule {
        rule_name: "family-crate-without-serving",
        source_crate: "jangtq2-qwen35",
        forbidden_crate: "model-serving",
        rationale: "the jangtq2 family crate must build and test without pulling in the serving crate, which consumes the family through a one-way edge",
    },
    // The model primitives (issue #2340) sit under every model family. An
    // edge back to the serving crate would let a shared kernel reach into one
    // family and would rebuild the primitives on every family edit, which the
    // split exists to stop.
    ForbiddenDependencyRule {
        rule_name: "model-primitives-without-serving",
        source_crate: "model-primitives",
        forbidden_crate: "model-serving",
        rationale: "the model primitives crate must build and test without pulling in the serving crate, which consumes the primitives through a one-way edge",
    },
    // The speech families (issue #2340) implement a seam the serving
    // crate's worker drives. An edge back would rebuild the serving crate on
    // every speech edit and let a family reach into the engine.
    ForbiddenDependencyRule {
        rule_name: "speech-without-serving",
        source_crate: "speech",
        forbidden_crate: "model-serving",
        rationale: "the speech crate must build and test without pulling in the serving crate, which drives the speech engines through a one-way edge",
    },
    // Each speech family is a crate of its own (issue #2473, Step 7). Like
    // the shell, it never reaches the serving crate, and one family never
    // reaches another, so a family can be added or changed alone.
    ForbiddenDependencyRule {
        rule_name: "kokoro-without-serving",
        source_crate: "kokoro",
        forbidden_crate: "model-serving",
        rationale: "the Kokoro crate must build and test without pulling in the serving crate, which drives the speech engines through a one-way edge",
    },
    ForbiddenDependencyRule {
        rule_name: "phonon-without-serving",
        source_crate: "phonon",
        forbidden_crate: "model-serving",
        rationale: "the Phonon crate must build and test without pulling in the serving crate, which drives the speech engines through a one-way edge",
    },
    ForbiddenDependencyRule {
        rule_name: "kokoro-without-phonon",
        source_crate: "kokoro",
        forbidden_crate: "phonon",
        rationale: "one speech family must not depend on another; what they share belongs in the speech shell",
    },
    ForbiddenDependencyRule {
        rule_name: "phonon-without-kokoro",
        source_crate: "phonon",
        forbidden_crate: "kokoro",
        rationale: "one speech family must not depend on another; what they share belongs in the speech shell",
    },
    // The image families (issue #2340), like speech, implement a seam the
    // serving crate's worker drives.
    ForbiddenDependencyRule {
        rule_name: "image-generation-without-serving",
        source_crate: "image-generation",
        forbidden_crate: "model-serving",
        rationale: "the image generation crate must build and test without pulling in the serving crate, which drives the image engines through a one-way edge",
    },
    // Artifact validation sits under every model family; an edge to the
    // serving crate would make each family depend on the engine through it.
    ForbiddenDependencyRule {
        rule_name: "artifact-validation-without-serving",
        source_crate: "artifact-validation",
        forbidden_crate: "model-serving",
        rationale: "artifact validation must build and test without pulling in the serving crate, because every model family depends on it",
    },
    // The engine core sits under every model family, which implements its
    // contract; an edge to the serving crate would make the split circular.
    ForbiddenDependencyRule {
        rule_name: "engine-core-without-serving",
        source_crate: "engine-core",
        forbidden_crate: "model-serving",
        rationale: "the engine core must build and test without pulling in the serving crate, because every model family builds on it",
    },
    // Family-neutral machinery moves out of the Qwen 3.5 family into the
    // engine core (issue #2473, Step 3), starting with the custom kernel
    // capability registry. An edge from the engine core back to the family
    // would let that machinery name a family type again.
    ForbiddenDependencyRule {
        rule_name: "engine-core-without-qwen35-family",
        source_crate: "engine-core",
        forbidden_crate: "qwen35",
        rationale: "the engine core holds machinery every model family shares, so it must build without any family crate; a family supplies its parts through the engine core's traits",
    },
    // The Qwen 3.5 family sits under the decision models and the serving
    // crate's worker; an edge back would make the split circular.
    ForbiddenDependencyRule {
        rule_name: "qwen35-without-serving",
        source_crate: "qwen35",
        forbidden_crate: "model-serving",
        rationale: "the Qwen 3.5 crate must build and test without pulling in the serving crate, which drives its engines through a one-way edge",
    },
    // The decision models sit above the Qwen 3.5 family and below the
    // serving crate's worker, which drives them.
    ForbiddenDependencyRule {
        rule_name: "decision-models-without-serving",
        source_crate: "decision-models",
        forbidden_crate: "model-serving",
        rationale: "the decision models crate must build and test without pulling in the serving crate, which drives the decision engines through a one-way edge",
    },
    // The quantizer reaches the Qwen 3.5 family only through the probe crate
    // above it (issue #2340), so the worker and daemon, which build on the
    // family, never compile the quantizer.
    ForbiddenDependencyRule {
        rule_name: "qwen35-without-quantizer",
        source_crate: "qwen35",
        forbidden_crate: "quantization",
        rationale: "the Qwen 3.5 crate must build without the quantizer, which its sensitivity probe crate brings in only for the CLI's convert verb and the acceptance tests",
    },
];

/// One detected violation with the offending dependency chain, source crate
/// first. The chain names every hop so a failure is attributable from the
/// message alone.
#[derive(Debug)]
pub(crate) struct ArchitectureViolation {
    pub(crate) rule_name: &'static str,
    pub(crate) rationale: &'static str,
    pub(crate) dependency_chain: Vec<String>,
}

pub fn run(repository_root: &Path) -> Result<(), String> {
    println!("[check-architecture] phase=repository-graph status=start");
    let metadata_json_text = workspace_metadata_json(repository_root)?;
    let mut violations = architecture_violations(&metadata_json_text)?;
    violations.extend(family_layer_violations(&metadata_json_text)?);
    report_verdict(&violations)
}

fn workspace_metadata_json(repository_root: &Path) -> Result<String, String> {
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
    Ok(String::from_utf8_lossy(&metadata_output.stdout).into_owned())
}

/// Prints one line per rule on success, or one line per violation, keeping
/// the output lines hooks and CI steps read.
fn report_verdict(violations: &[ArchitectureViolation]) -> Result<(), String> {
    if !violations.is_empty() {
        for violation in violations {
            eprintln!(
                "[check-architecture] rule={} violation: {}; {}",
                violation.rule_name,
                violation.dependency_chain.join(" -> "),
                violation.rationale
            );
        }
        return Err(format!(
            "[check-architecture] status=failed violations={}",
            violations.len()
        ));
    }
    for rule in FORBIDDEN_DEPENDENCY_RULES {
        println!("[check-architecture] rule={} status=pass", rule.rule_name);
    }
    for family_rule_name in FAMILY_LAYER_RULE_NAMES {
        println!("[check-architecture] rule={family_rule_name} status=pass");
    }
    println!(
        "[check-architecture] status=success rules={}",
        FORBIDDEN_DEPENDENCY_RULES.len() + FAMILY_LAYER_RULE_NAMES.len()
    );
    Ok(())
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
pub(crate) fn dependency_graph_from_metadata(
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
pub(crate) fn shortest_dependency_chain(
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
