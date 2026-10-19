//! Typed port of `scripts/ci/report-tier-pyramid.sh`: counts tests per tier
//! statically so any laptop reproduces the distribution, compares it with the
//! committed baseline at a drift tolerance, and can re-record the baseline.
//!
//! Counting parity notes (must match the historical numbers exactly):
//! - unit: lines that are exactly `#[test]` or that start with `#[tokio::test`
//!   at column zero, under beside-source trees.
//! - integration: the same unindented `#[test]` and `#[tokio::test`
//!   attribute lines under `tests/` directories, excluding helper folders.
//! - e2e / acceptance split of the root e2e package: files with the
//!   `ASTRONOMICAL_REAL_MODEL` gate count as acceptance.
//! - acceptance also counts the root acceptance package.

use std::fs;
use std::path::{Path, PathBuf};

const DRIFT_TOLERANCE_PERCENT: i64 = 5;
const BASELINE_RELATIVE_PATH: &str = "scripts/ci/tier-pyramid-baseline.json";

fn is_test_attribute_line(line: &str) -> bool {
    line == "#[test]" || line.starts_with("#[tokio::test")
}

fn count_matching_lines(source_file: &Path) -> usize {
    let Ok(source_text) = fs::read_to_string(source_file) else {
        return 0;
    };
    source_text
        .lines()
        .filter(|line| is_test_attribute_line(line))
        .count()
}

fn collect_rs_files(directory: &Path, recursive: bool) -> Vec<PathBuf> {
    let mut collected = Vec::new();
    let Ok(entries) = fs::read_dir(directory) else {
        return collected;
    };
    for entry in entries.flatten() {
        let entry_path = entry.path();
        if entry_path.is_dir() {
            if recursive {
                collected.extend(collect_rs_files(&entry_path, true));
            }
        } else if entry_path
            .extension()
            .is_some_and(|extension| extension == "rs")
        {
            collected.push(entry_path);
        }
    }
    collected
}

fn excluded_helper_directory(relative_path: &str, helper_names: &[&str]) -> bool {
    helper_names
        .iter()
        .any(|helper_name| relative_path.contains(&format!("/{helper_name}/")))
}

fn count_unit_tier(repository_root: &Path) -> usize {
    let mut total = 0;
    for package_root in glob_package_source_roots(repository_root) {
        for source_file in collect_rs_files(&package_root, true) {
            total += count_matching_lines(&source_file);
        }
    }
    total
}

fn glob_package_source_roots(repository_root: &Path) -> Vec<PathBuf> {
    let mut roots = Vec::new();
    for base in ["crates", "apps"] {
        let Ok(entries) = fs::read_dir(repository_root.join(base)) else {
            continue;
        };
        for entry in entries.flatten() {
            let source_root = entry.path().join("src");
            if source_root.is_dir() {
                roots.push(source_root);
            }
        }
    }
    roots
}

fn glob_package_test_roots(repository_root: &Path) -> Vec<PathBuf> {
    let mut roots = Vec::new();
    for base in ["crates", "apps"] {
        let Ok(entries) = fs::read_dir(repository_root.join(base)) else {
            continue;
        };
        for entry in entries.flatten() {
            let test_root = entry.path().join("tests");
            if test_root.is_dir() {
                roots.push(test_root);
            }
        }
    }
    roots
}

fn count_integration_tier(repository_root: &Path) -> usize {
    let mut total = 0;
    for test_root in glob_package_test_roots(repository_root) {
        for source_file in collect_rs_files(&test_root, true) {
            let relative_path = source_file
                .strip_prefix(repository_root)
                .unwrap_or(&source_file)
                .to_string_lossy()
                .replace('\\', "/");
            if excluded_helper_directory(&relative_path, &["e2e", "fixtures", "support", "common"])
            {
                continue;
            }
            total += count_matching_lines(&source_file);
        }
    }
    total
}

fn file_contains_literal(source_file: &Path, literal: &str) -> bool {
    fs::read_to_string(source_file)
        .map(|source_text| source_text.contains(literal))
        .unwrap_or(false)
}

fn count_e2e_package_tier(repository_root: &Path, acceptance_files: bool) -> usize {
    let e2e_tests_directory = repository_root.join("tests/e2e/tests");
    if !e2e_tests_directory.is_dir() {
        return 0;
    }
    let mut total = 0;
    for source_file in collect_rs_files(&e2e_tests_directory, true) {
        let relative_path = source_file
            .strip_prefix(repository_root)
            .unwrap_or(&source_file)
            .to_string_lossy()
            .replace('\\', "/");
        if excluded_helper_directory(&relative_path, &["common", "fixtures", "support"]) {
            continue;
        }
        let has_real_model_toggle = file_contains_literal(&source_file, "ASTRONOMICAL_REAL_MODEL");
        if has_real_model_toggle != acceptance_files {
            continue;
        }
        total += count_matching_lines(&source_file);
    }
    total
}

fn count_acceptance_package_tier(repository_root: &Path) -> usize {
    let acceptance_tests_directory = repository_root.join("tests/acceptance/tests");
    if !acceptance_tests_directory.is_dir() {
        return 0;
    }
    let mut total = 0;
    for source_file in collect_rs_files(&acceptance_tests_directory, true) {
        let relative_path = source_file
            .strip_prefix(repository_root)
            .unwrap_or(&source_file)
            .to_string_lossy()
            .replace('\\', "/");
        if excluded_helper_directory(&relative_path, &["fixtures", "support", "common"]) {
            continue;
        }
        total += count_matching_lines(&source_file);
    }
    total
}

fn read_baseline(repository_root: &Path) -> Option<[usize; 4]> {
    let baseline_text = fs::read_to_string(repository_root.join(BASELINE_RELATIVE_PATH)).ok()?;
    let mut tiers = [0usize; 4];
    for (field_index, field_name) in ["unit", "integration", "e2e", "acceptance"]
        .into_iter()
        .enumerate()
    {
        let marker = format!("\"{field_name}\": ");
        let value_start = baseline_text.find(&marker)? + marker.len();
        let value_end = baseline_text[value_start..]
            .find(|character: char| !character.is_ascii_digit())
            .map(|offset| value_start + offset)
            .unwrap_or(baseline_text.len());
        tiers[field_index] = baseline_text[value_start..value_end].parse().ok()?;
    }
    Some(tiers)
}

fn write_baseline(repository_root: &Path, tiers: [usize; 4]) -> Result<(), String> {
    let baseline_path = repository_root.join(BASELINE_RELATIVE_PATH);
    if let Some(parent_directory) = baseline_path.parent() {
        fs::create_dir_all(parent_directory).map_err(|error| {
            format!("[tier-pyramid] status=failed reason=baseline-directory error={error}")
        })?;
    }
    let baseline_text = format!(
        "{{\n  \"unit\": {},\n  \"integration\": {},\n  \"e2e\": {},\n  \"acceptance\": {}\n}}\n",
        tiers[0], tiers[1], tiers[2], tiers[3]
    );
    fs::write(&baseline_path, baseline_text).map_err(|error| {
        format!(
            "[tier-pyramid] status=failed reason=baseline-write path={} error={error}",
            baseline_path.display()
        )
    })
}

pub fn run(repository_root: &Path, update_baseline: bool) -> Result<(), String> {
    let tiers = [
        count_unit_tier(repository_root),
        count_integration_tier(repository_root),
        count_e2e_package_tier(repository_root, false),
        count_e2e_package_tier(repository_root, true)
            + count_acceptance_package_tier(repository_root),
    ];
    let total: usize = tiers.iter().sum();

    println!("[tier-pyramid] status=start mode=report");
    for (tier_name, tier_count) in ["unit", "integration", "e2e", "acceptance"]
        .into_iter()
        .zip(tiers)
    {
        let share = if total > 0 {
            tier_count as f64 * 100.0 / total as f64
        } else {
            0.0
        };
        println!("{tier_name}={tier_count} share={share:.1}%");
    }
    println!("total={total}");

    if update_baseline {
        write_baseline(repository_root, tiers)?;
        println!("[tier-pyramid] status=baseline-written path={BASELINE_RELATIVE_PATH}");
        println!("[tier-pyramid] status=success");
        return Ok(());
    }

    let Some(baseline) = read_baseline(repository_root) else {
        println!("[tier-pyramid] status=no-baseline hint=record-one-with---update-baseline");
        println!("[tier-pyramid] status=success");
        return Ok(());
    };
    let baseline_total: usize = baseline.iter().sum();
    if baseline_total == 0 || total == 0 {
        println!("[tier-pyramid] status=empty-distribution hint=record-one-with---update-baseline");
        println!("[tier-pyramid] status=success");
        return Ok(());
    }

    for (tier_index, tier_name) in ["unit", "integration", "e2e", "acceptance"]
        .into_iter()
        .enumerate()
    {
        let live_share = tiers[tier_index] as f64 * 100.0 / total as f64;
        let baseline_share = baseline[tier_index] as f64 * 100.0 / baseline_total as f64;
        if (live_share - baseline_share).abs() > DRIFT_TOLERANCE_PERCENT as f64 {
            println!(
                "[tier-pyramid] status=drift-detected tier={tier_name} drifted {:.1} percentage points from the recorded baseline",
                (live_share - baseline_share).abs()
            );
            println!(
                "[tier-pyramid] hint=if-this-migration-is-intended-refresh-{BASELINE_RELATIVE_PATH}-with---update-baseline-in-a-reviewed-pull-request"
            );
            return Err(String::new());
        }
    }
    println!("[tier-pyramid] status=success reason=within-tolerance");
    Ok(())
}
