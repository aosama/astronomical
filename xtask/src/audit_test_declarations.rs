//! Typed port of the manual orphaned-test-file hunt: every `.rs` file under a
//! crate's `tests/` tree must be reachable from a cargo test target through a
//! chain of `mod` declarations, or it is dead code that never compiles and
//! never runs. The audit exists because `tests/integration/`
//! `flux2_klein_reference_oracle.rs` sat undeclared for its whole life: seven
//! passing-looking tests that no runner ever executed.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Reports progress on the same cadence as the other xtask scans: the
/// repository-wide walk finishes in well under a second, so the five-second
/// reporting contract is satisfied by the start and completion lines with a
/// zero-second estimate.
const PROGRESS_CRATE_INTERVAL: usize = 25;

/// One finding: a test-tree file or directory no cargo target can reach.
///
/// `kind` names what is unreachable ("file" or "directory"), `crate_name` the
/// owning crate, and `path` the crate-relative location under `tests/`.
pub(crate) struct OrphanFinding {
    kind: &'static str,
    crate_name: String,
    path: String,
}

impl OrphanFinding {
    pub(crate) fn describe(&self) -> String {
        format!(
            "orphaned test {}: {} ({}) — no `mod` declaration reaches it, so it never compiles and never runs",
            self.kind, self.path, self.crate_name
        )
    }
}

/// The pure audit core: values in, findings out, no filesystem.
///
/// `test_sources` maps crate-relative paths of every `.rs` file under a
/// crate's `tests/` tree to its content; `manifest_target_paths` holds the
/// crate-relative `[[bin]]`/`[[test]]` target paths from the crate's manifest.
///
/// The audit walks the module graph: roots are the cargo-discovered test
/// targets (`tests/<name>.rs`) plus the manifest's fixture binaries, and an
/// edge exists wherever one file declares a `mod` whose resolution lands on
/// another file in the tree — the standard `mod name;` lookup, or the
/// `#[path = "..."] mod name;` form that relocates the file (the four wiring
/// styles this repository uses all reduce to those two edges). Every
/// unreachable file is an orphan: it never compiles and never runs.
pub(crate) fn audit_test_declarations(
    crate_name: &str,
    test_sources: &BTreeMap<String, String>,
    manifest_target_paths: &BTreeSet<String>,
) -> Vec<OrphanFinding> {
    let is_manifest_target =
        |path: &str| manifest_target_paths.contains(&path.replace('\\', "/").replace("./", ""));

    // Roots: cargo auto-discovers root-level `tests/<name>.rs` entries as test
    // targets, and manifest `[[bin]] path` sources as fixture binaries.
    let mut reachable: BTreeSet<String> = BTreeSet::new();
    for path in test_sources.keys() {
        let is_root_target = path
            .strip_prefix("tests/")
            .is_some_and(|remainder| remainder.ends_with(".rs") && !remainder.contains('/'));
        if is_root_target || is_manifest_target(path) {
            reachable.insert(path.clone());
        }
    }

    // Edges: every `mod` declaration in a reached file, resolved the way
    // rustc resolves it.
    let mut frontier: Vec<String> = reachable.iter().cloned().collect();
    while let Some(containing_path) = frontier.pop() {
        let containing_directory = parent_directory(&containing_path);
        let Some(containing_content) = test_sources.get(&containing_path) else {
            continue;
        };
        // rustc looks a declared module up in the containing file's directory
        // and — for a named file module — in the sibling folder the file owns
        // (launch.rs owns launch/). A mod.rs owns its own directory.
        let mut lookup_prefixes: Vec<String> = Vec::new();
        if let Some(directory) = &containing_directory {
            lookup_prefixes.push(directory.clone());
            let is_mod_rs = containing_path.ends_with("/mod.rs");
            let file_stem = containing_path
                .rsplit('/')
                .next()
                .and_then(|name| name.strip_suffix(".rs"));
            if !is_mod_rs
                && let Some(file_stem) = file_stem
            {
                let owned_folder = format!("{directory}/{file_stem}");
                if !lookup_prefixes.contains(&owned_folder) {
                    lookup_prefixes.push(owned_folder);
                }
            }
        }
        for (declared_module_name, relocated_path) in module_declarations(containing_content) {
            let resolutions: Vec<String> = match relocated_path {
                Some(relocated_path) => {
                    let base = containing_directory.clone().unwrap_or_default();
                    vec![normalize_module_path(&format!("{base}/{relocated_path}"))]
                }
                None => lookup_prefixes
                    .iter()
                    .flat_map(|prefix| {
                        [
                            normalize_module_path(&format!("{prefix}/{declared_module_name}.rs")),
                            normalize_module_path(&format!(
                                "{prefix}/{declared_module_name}/mod.rs"
                            )),
                        ]
                    })
                    .collect(),
            };
            for resolution in resolutions {
                if test_sources.contains_key(&resolution) && reachable.insert(resolution.clone()) {
                    frontier.push(resolution);
                }
            }
        }
    }

    test_sources
        .keys()
        .filter(|path| !reachable.contains(*path))
        .map(|path| OrphanFinding {
            kind: "file",
            crate_name: crate_name.to_owned(),
            path: path.clone(),
        })
        .collect()
}

/// Extracts the `(module_name, Option<relocated_path>)` pairs a file declares.
///
/// Attributes may interleave freely before their `mod` — the acceptance suite
/// puts `#[cfg(feature = ...)]` between `#[path = ...]` and `mod` — so a path
/// attribute stays pending until the next declaration consumes it. Any other
/// statement breaks the adjacency, matching how rustc scopes attributes.
fn module_declarations(file_content: &str) -> Vec<(String, Option<String>)> {
    let mut declarations = Vec::new();
    let mut pending_path: Option<String> = None;
    for line in file_content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with("//") {
            continue;
        }
        if trimmed.starts_with("#[") {
            if trimmed.starts_with("#[path") {
                pending_path = extract_quoted_path(trimmed);
            }
            continue;
        }
        let Some(module_name) = declared_module_name(trimmed) else {
            pending_path = None;
            continue;
        };
        declarations.push((module_name, pending_path.take()));
    }
    declarations
}

/// Parses one already-trimmed line as `mod <name>;` (optionally visibility-
/// qualified) and returns the module name.
fn declared_module_name(trimmed_line: &str) -> Option<String> {
    if !trimmed_line.ends_with(';') {
        return None;
    }
    let without_semicolon = &trimmed_line[..trimmed_line.len() - 1];
    let name = if let Some(rest) = without_semicolon.strip_prefix("mod ") {
        rest
    } else {
        without_semicolon.split_whitespace().last()?
    };
    let name = name.trim();
    let is_bare_identifier = !name.is_empty()
        && name
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || character == '_')
        && !name.chars().next().is_some_and(|c| c.is_ascii_digit());
    is_bare_identifier.then(|| name.to_owned())
}

/// Lexically resolves `.` and `..` segments in a module path.
fn normalize_module_path(path: &str) -> String {
    let mut resolved_segments: Vec<&str> = Vec::new();
    for segment in path.split('/') {
        match segment {
            "." | "" => {}
            ".." => {
                resolved_segments.pop();
            }
            other => resolved_segments.push(other),
        }
    }
    resolved_segments.join("/")
}

fn parent_directory(path: &str) -> Option<String> {
    path.rsplit_once('/').map(|(parent, _)| parent.to_owned())
}

fn extract_quoted_path(line: &str) -> Option<String> {
    let value_start = line.find('"')? + 1;
    let value_end = line[value_start..].find('"')? + value_start;
    Some(line[value_start..value_end].to_owned())
}

/// Reads the crate-relative target paths out of a manifest's `[[bin]]` and
/// `[[test]]` sections. Fixture binaries (`tests/fixtures/*.rs`) and explicitly
/// named test targets (`[[test]] path = "tests/gpu_contracts/direct_mlx.rs"`)
/// are cargo entry points at arbitrary depths — they are roots, not modules.
pub(crate) fn target_paths_from_manifest(manifest_content: &str) -> BTreeSet<String> {
    let mut target_paths = BTreeSet::new();
    let mut inside_target_section = false;
    for line in manifest_content.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') {
            inside_target_section = trimmed == "[[bin]]" || trimmed == "[[test]]";
            continue;
        }
        if inside_target_section
            && trimmed.starts_with("path")
            && let Some(target_path) = extract_quoted_path(trimmed)
        {
            target_paths.insert(target_path);
        }
    }
    target_paths
}

fn repository_test_crate_manifests(repository_root: &Path) -> Vec<PathBuf> {
    let mut manifest_paths = Vec::new();
    let mut pending_directories = vec![repository_root.to_path_buf()];
    while let Some(directory) = pending_directories.pop() {
        let Ok(directory_entries) = fs::read_dir(&directory) else {
            continue;
        };
        for entry in directory_entries.flatten() {
            let entry_path = entry.path();
            let entry_name = entry.file_name().to_string_lossy().to_string();
            if !entry_path.is_dir() {
                continue;
            }
            if entry_name == "target" || entry_name.starts_with('.') {
                continue;
            }
            let manifest_path = entry_path.join("Cargo.toml");
            if manifest_path.is_file() {
                manifest_paths.push(manifest_path);
            }
            pending_directories.push(entry_path);
        }
    }
    manifest_paths.sort();
    manifest_paths
}

fn collect_test_sources(crate_root: &Path) -> BTreeMap<String, String> {
    let mut test_sources = BTreeMap::new();
    let tests_root = crate_root.join("tests");
    let mut pending_directories = vec![tests_root.clone()];
    while let Some(directory) = pending_directories.pop() {
        let Ok(directory_entries) = fs::read_dir(&directory) else {
            continue;
        };
        for entry in directory_entries.flatten() {
            let entry_path = entry.path();
            if entry_path.is_dir() {
                pending_directories.push(entry_path);
                continue;
            }
            if entry_path
                .extension()
                .is_none_or(|extension| extension != "rs")
            {
                continue;
            }
            let Ok(source_content) = fs::read_to_string(&entry_path) else {
                continue;
            };
            let Ok(relative_path) = entry_path.strip_prefix(crate_root) else {
                continue;
            };
            test_sources.insert(
                relative_path.to_string_lossy().replace('\\', "/"),
                source_content,
            );
        }
    }
    test_sources
}

pub(crate) fn run(repository_root: &Path) -> Result<(), String> {
    let started_at = Instant::now();
    let manifest_paths = repository_test_crate_manifests(repository_root);
    let total_manifests = manifest_paths.len();
    println!(
        "[test-declarations] status=start crates={total_manifests} root={}",
        repository_root.display()
    );

    let mut findings = Vec::new();
    for (checked_manifests, manifest_path) in manifest_paths.iter().enumerate() {
        let crate_root = manifest_path
            .parent()
            .ok_or_else(|| format!("manifest without parent: {}", manifest_path.display()))?;
        let crate_name = crate_root
            .file_name()
            .map(|name| name.to_string_lossy().to_string())
            .unwrap_or_else(|| "<unknown>".to_owned());
        let test_sources = collect_test_sources(crate_root);
        if test_sources.is_empty() {
            continue;
        }
        let manifest_content = fs::read_to_string(manifest_path)
            .map_err(|error| format!("{}: {error}", manifest_path.display()))?;
        let manifest_target_paths = target_paths_from_manifest(&manifest_content);
        findings.extend(audit_test_declarations(
            &crate_name,
            &test_sources,
            &manifest_target_paths,
        ));

        let checked_manifests = checked_manifests + 1;
        let elapsed_seconds = started_at.elapsed().as_secs();
        let percentage = checked_manifests * 100 / total_manifests.max(1);
        let eta_seconds = if elapsed_seconds == 0 {
            0
        } else {
            (elapsed_seconds * total_manifests as u64) / checked_manifests as u64 - elapsed_seconds
        };
        if checked_manifests % PROGRESS_CRATE_INTERVAL == 0 || checked_manifests == total_manifests
        {
            println!(
                "[test-declarations] status=progress crates={checked_manifests}/{total_manifests} percentage={percentage} ETA_seconds={eta_seconds} elapsed_seconds={elapsed_seconds}"
            );
        }
    }

    if findings.is_empty() {
        println!(
            "[test-declarations] status=success orphans=0 elapsed_seconds={}",
            started_at.elapsed().as_secs()
        );
        return Ok(());
    }
    for finding in &findings {
        println!("{}", finding.describe());
    }
    println!(
        "[test-declarations] status=failed orphans={} elapsed_seconds={}",
        findings.len(),
        started_at.elapsed().as_secs()
    );
    Err(format!(
        "{} orphaned test declaration{} found; every file under a crate's tests/ tree must be reachable from a test target through `mod` declarations",
        findings.len(),
        if findings.len() == 1 { "" } else { "s" }
    ))
}
