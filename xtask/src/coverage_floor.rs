//! Pre-commit coverage floor: fails a local commit when a package's
//! llvm-cov line coverage falls below the configured floor.
//!
//! This gate is deliberately local-only (`.githooks/pre-commit`); it is not
//! a continuous-integration step, so a cold instrumented build never slows a
//! pull request. The command runs cargo-llvm-cov with the package's offline
//! test scope, parses the summary `TOTAL` row, and compares line coverage
//! against the floor, failing closed on any parsing or tooling failure.

use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// The package whose coverage this gate protects by default: the pure
/// admission, eviction, and residency arithmetic every family trusts.
pub const DEFAULT_PACKAGE_NAME: &str = "astronomical-memory-policy";
pub const DEFAULT_MINIMUM_LINE_COVERAGE_PERCENT: f64 = 90.0;
const HEARTBEAT_INTERVAL_SECONDS: u64 = 30;
const MAXIMUM_ELAPSED_SECONDS: u64 = 900;

/// One parsed `TOTAL` row of a cargo-llvm-cov summary table.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CoverageSummary {
    pub region_coverage_percent: f64,
    pub function_coverage_percent: f64,
    pub line_coverage_percent: f64,
}

impl CoverageSummary {
    /// A coverage at exactly the floor passes; only strictly-below fails.
    pub fn is_below_floor(&self, minimum_line_coverage_percent: f64) -> bool {
        self.line_coverage_percent + f64::EPSILON < minimum_line_coverage_percent
    }
}

/// The parsed command line of `cargo xtask coverage-floor`.
#[derive(Clone, Debug, PartialEq)]
pub struct CoverageFloorInvocation {
    pub package_name: String,
    pub minimum_line_coverage_percent: f64,
}

impl CoverageFloorInvocation {
    pub fn parse(arguments: &[String]) -> Result<Self, String> {
        let mut invocation = Self {
            package_name: DEFAULT_PACKAGE_NAME.to_owned(),
            minimum_line_coverage_percent: DEFAULT_MINIMUM_LINE_COVERAGE_PERCENT,
        };
        let mut argument_index = 0;
        while argument_index < arguments.len() {
            let argument = arguments[argument_index].as_str();
            let value = arguments.get(argument_index + 1).ok_or_else(|| {
                format!(
                    "[coverage-floor] status=failed reason=missing-value-for argument={argument}"
                )
            })?;
            match argument {
                "--package" => {
                    invocation.package_name = value.clone();
                }
                "--minimum-percent" => {
                    invocation.minimum_line_coverage_percent = value
                        .parse()
                        .map_err(|_| {
                            format!(
                                "[coverage-floor] status=failed reason=non-numeric-percent value={value}"
                            )
                        })
                        .and_then(|parsed_percent| {
                            if (0.0..=100.0).contains(&parsed_percent) {
                                Ok(parsed_percent)
                            } else {
                                Err(format!(
                                    "[coverage-floor] status=failed reason=percent-out-of-range value={value}"
                                ))
                            }
                        })?;
                }
                unexpected => {
                    return Err(format!(
                        "[coverage-floor] status=failed reason=unknown-argument argument={unexpected}"
                    ));
                }
            }
            argument_index += 2;
        }
        Ok(invocation)
    }
}

/// One parsed percentage cell such as `88.17%`.
fn parse_percent_cell(percent_cell: &str) -> Result<f64, String> {
    percent_cell
        .strip_suffix('%')
        .ok_or_else(|| {
            format!(
                "[coverage-floor] status=failed reason=malformed-percent-cell cell={percent_cell}"
            )
        })?
        .parse()
        .map_err(|_| {
            format!(
                "[coverage-floor] status=failed reason=malformed-percent-cell cell={percent_cell}"
            )
        })
}

/// Extracts the `TOTAL` row percentages from a cargo-llvm-cov summary table.
pub fn parse_summary_total_row(summary_text: &str) -> Result<CoverageSummary, String> {
    let total_row = summary_text
        .lines()
        .rev()
        .find(|line| line.starts_with("TOTAL"))
        .ok_or_else(|| "[coverage-floor] status=failed reason=no-total-row".to_owned())?;
    let cells: Vec<&str> = total_row.split_whitespace().collect();
    // Column layout: TOTAL, regions, missed, regions%, functions, missed,
    // functions%, lines, missed, lines%, branches, missed, branches%.
    if cells.len() < 10 {
        return Err(format!(
            "[coverage-floor] status=failed reason=short-total-row columns={}",
            cells.len()
        ));
    }
    Ok(CoverageSummary {
        region_coverage_percent: parse_percent_cell(cells[3])?,
        function_coverage_percent: parse_percent_cell(cells[6])?,
        line_coverage_percent: parse_percent_cell(cells[9])?,
    })
}

pub fn run(repository_root: &Path, arguments: &[String]) -> Result<(), String> {
    let invocation = CoverageFloorInvocation::parse(arguments)?;
    println!(
        "[coverage-floor] status=start package={} floor_percent={:.2} timeout_seconds={} scope=\"--lib --test offline_tests\"",
        invocation.package_name, invocation.minimum_line_coverage_percent, MAXIMUM_ELAPSED_SECONDS
    );
    let started_at = Instant::now();
    let coverage_summary = measure_line_coverage(repository_root, &invocation, started_at)?;
    println!(
        "[coverage-floor] status=measured regions_cover={:.2}% functions_cover={:.2}% lines_cover={:.2}%",
        coverage_summary.region_coverage_percent,
        coverage_summary.function_coverage_percent,
        coverage_summary.line_coverage_percent
    );
    if coverage_summary.is_below_floor(invocation.minimum_line_coverage_percent) {
        return Err(format!(
            "[coverage-floor] status=failed package={} line_coverage_percent={:.2} minimum_percent={:.2} reason=below-floor",
            invocation.package_name,
            coverage_summary.line_coverage_percent,
            invocation.minimum_line_coverage_percent
        ));
    }
    println!(
        "[coverage-floor] status=success package={} line_coverage_percent={:.2} minimum_percent={:.2} elapsed_seconds={}",
        invocation.package_name,
        coverage_summary.line_coverage_percent,
        invocation.minimum_line_coverage_percent,
        started_at.elapsed().as_secs()
    );
    Ok(())
}

/// Removes the summary scratch file on every exit path — early errors
/// included — so repeated runs never strand files in the temp directory.
struct SummaryFileGuard(std::path::PathBuf);

impl Drop for SummaryFileGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn measure_line_coverage(
    repository_root: &Path,
    invocation: &CoverageFloorInvocation,
    started_at: Instant,
) -> Result<CoverageSummary, String> {
    // The summary lands on stdout; a file keeps the pipe from filling while
    // the heartbeat polls. Stderr stays inherited so cargo's own live
    // compile and test progress streams to the committer during the run.
    let summary_file_path = std::env::temp_dir().join(format!(
        "astronomical-xtask-coverage-floor-{}.summary",
        std::process::id()
    ));
    let summary_file = File::create(&summary_file_path).map_err(|error| {
        format!("[coverage-floor] status=failed reason=summary-file error={error}")
    })?;
    // The underscore name keeps rustc quiet: the guard is never read again,
    // and its Drop — which removes the scratch file — still runs on every
    // exit path of this function.
    let _summary_file_guard = SummaryFileGuard(summary_file_path.clone());
    let mut coverage_child_command = Command::new("cargo");
    coverage_child_command
        .arg("llvm-cov")
        .arg("-p")
        .arg(&invocation.package_name)
        .arg("--lib")
        .arg("--test")
        .arg("offline_tests")
        .arg("--summary-only")
        .current_dir(repository_root)
        .stdout(Stdio::from(summary_file))
        .stderr(Stdio::inherit());
    // sccache cache hits were built without coverage instrumentation, so a
    // hit would silently understate coverage and could fail this gate on a
    // warm cache: measure with an unwrapped rustc instead.
    coverage_child_command
        .env_remove("RUSTC_WRAPPER")
        .env_remove("CARGO_BUILD_RUSTC_WRAPPER");
    attach_own_process_group(&mut coverage_child_command);
    let mut coverage_child = coverage_child_command
        .spawn()
        .map_err(|error| {
            format!(
                "[coverage-floor] status=failed reason=spawn-failed hint=\"cargo install cargo-llvm-cov --locked && rustup component add llvm-tools-preview\" error={error}"
            )
        })?;
    let mut last_heartbeat_seconds = 0_u64;
    loop {
        match coverage_child.try_wait() {
            Ok(Some(exit_status)) => {
                if !exit_status.success() {
                    return Err(format!(
                        "[coverage-floor] status=failed package={} reason=cargo-llvm-cov-exited exit_status={exit_status}",
                        invocation.package_name
                    ));
                }
                break;
            }
            Ok(None) => {}
            Err(error) => {
                // The child may still be alive when try_wait itself fails;
                // take the whole tree down before failing closed.
                terminate_process_tree(&mut coverage_child);
                return Err(format!(
                    "[coverage-floor] status=failed reason=wait error={error}"
                ));
            }
        }
        let elapsed_seconds = started_at.elapsed().as_secs();
        if elapsed_seconds >= MAXIMUM_ELAPSED_SECONDS {
            terminate_process_tree(&mut coverage_child);
            return Err(format!(
                "[coverage-floor] status=failed reason=timeout limit_seconds={MAXIMUM_ELAPSED_SECONDS}"
            ));
        }
        if elapsed_seconds.saturating_sub(last_heartbeat_seconds) >= HEARTBEAT_INTERVAL_SECONDS {
            println!(
                "[coverage-floor] status=progress phase=measure elapsed_seconds={elapsed_seconds} remaining_seconds={}",
                MAXIMUM_ELAPSED_SECONDS.saturating_sub(elapsed_seconds)
            );
            last_heartbeat_seconds = elapsed_seconds;
        }
        std::thread::sleep(Duration::from_secs(1));
    }
    let mut summary_text = String::new();
    File::open(&summary_file_path)
        .and_then(|mut opened_file| opened_file.read_to_string(&mut summary_text))
        .map_err(|error| {
            format!("[coverage-floor] status=failed reason=summary-read error={error}")
        })?;
    // The guard outlives this read and removes the scratch file on return,
    // whether parsing succeeds or fails.
    parse_summary_total_row(&summary_text)
}

/// Runs the coverage child in its own process group on Unix so a timeout can
/// take down cargo together with its rustc and test grandchildren; a bare
/// `Child::kill` signals only the cargo parent and would orphan the tree.
#[cfg(unix)]
fn attach_own_process_group(coverage_child_command: &mut Command) {
    use std::os::unix::process::CommandExt;
    // Process group 0 puts the child at the head of a fresh group whose
    // identifier equals the child's process identifier.
    coverage_child_command.process_group(0);
}

#[cfg(not(unix))]
fn attach_own_process_group(_coverage_child_command: &mut Command) {}

#[cfg(unix)]
fn terminate_process_tree(coverage_child: &mut std::process::Child) {
    let process_group_id = coverage_child.id();
    let group_kill_succeeded = Command::new("kill")
        .arg("-9")
        .arg(format!("-{process_group_id}"))
        .status()
        .is_ok_and(|kill_status| kill_status.success());
    if !group_kill_succeeded {
        let _ = coverage_child.kill();
    }
}

#[cfg(not(unix))]
fn terminate_process_tree(coverage_child: &mut std::process::Child) {
    let _ = coverage_child.kill();
}
