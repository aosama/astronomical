//! `cargo xtask check-duplication` — the duplication ratchet of issue #2473,
//! step 0, over `duplication-baseline.json`.
//!
//! It measures repeated blocks: a code line counts as repeated when it sits
//! in a run of at least six consecutive code lines that also appears in
//! another file, after blank lines and `//` comment lines are dropped,
//! whitespace is trimmed and model-family names are masked (see
//! [`source_normalization`]). Production code and test code are measured as
//! two corpora that are never compared with each other, and each has its own
//! count in the baseline. Either count may stay or go down, never up.
//!
//! Like the CRAP ratchet, the tree is judged against a committed baseline,
//! never the working-tree copy: `HEAD`'s by default, as the pre-commit hook
//! runs it, or another ref's with `--against REF`, as the pre-push hook runs
//! it against `origin/main`. A commit therefore cannot raise the baseline and
//! add the copy together. `--write` replaces the baseline with the measured
//! counts, and only after that check passes, so the file can only go down.
//!
//! The measure undercounts on purpose: it sees only text that repeats line
//! for line. Two copies that do the same job with different variable names
//! are invisible to it, so a falling count is evidence of progress, not proof
//! that no copies remain.

pub(crate) mod duplication_baseline;
pub(crate) mod progress;
pub(crate) mod repeated_blocks;
pub(crate) mod report;
pub(crate) mod source_classification;
pub(crate) mod source_normalization;
pub(crate) mod source_walk;

use std::path::Path;
use std::process::Command;
use std::time::Instant;

use duplication_baseline::{
    BASELINE_RELATIVE_PATH, RepeatedLineCounts, has_grown_corpus, is_absent_at_ref_message,
    judge_committed_baseline, render_baseline,
};
use repeated_blocks::{DuplicationMeasurement, measure_repeated_blocks};

const DEFAULT_BASELINE_REF: &str = "HEAD";
const DEFAULT_TOP_PAIR_COUNT: usize = 15;

/// The parsed command line of `cargo xtask check-duplication`.
#[derive(Debug, PartialEq)]
pub(crate) struct DuplicationInvocation {
    pub(crate) should_write_baseline: bool,
    /// The git ref whose committed baseline the tree is judged against.
    pub(crate) baseline_ref: String,
    /// How many area pairs and file pairs to print for each corpus.
    pub(crate) top_pair_count: usize,
}

impl Default for DuplicationInvocation {
    fn default() -> Self {
        Self {
            should_write_baseline: false,
            baseline_ref: DEFAULT_BASELINE_REF.to_owned(),
            top_pair_count: DEFAULT_TOP_PAIR_COUNT,
        }
    }
}

impl DuplicationInvocation {
    pub(crate) fn parse(arguments: &[String]) -> Result<Self, String> {
        let mut invocation = Self::default();
        let mut remaining_arguments = arguments.iter();
        while let Some(argument) = remaining_arguments.next() {
            match argument.as_str() {
                "--write" => invocation.should_write_baseline = true,
                "--against" => {
                    invocation.baseline_ref =
                        take_flag_value("--against", &mut remaining_arguments)?;
                }
                "--top" => {
                    let top_text = take_flag_value("--top", &mut remaining_arguments)?;
                    invocation.top_pair_count = top_text.parse().map_err(|_| {
                        format!("[check-duplication] status=failed reason=top-is-not-a-count value={top_text}")
                    })?;
                }
                unknown_argument => {
                    return Err(format!(
                        "[check-duplication] status=failed reason=unknown-argument argument={unknown_argument} hint=\"the arguments are --write, --against REF and --top N\""
                    ));
                }
            }
        }
        Ok(invocation)
    }
}

/// The value after a flag. A value starting with `-` is another flag, not a
/// value, so `--against --write` is refused rather than read as a ref.
fn take_flag_value<'argument>(
    flag_name: &str,
    remaining_arguments: &mut impl Iterator<Item = &'argument String>,
) -> Result<String, String> {
    match remaining_arguments.next() {
        Some(flag_value) if !flag_value.starts_with('-') => Ok(flag_value.clone()),
        _ => Err(format!(
            "[check-duplication] status=failed reason=missing-value-for argument={flag_name}"
        )),
    }
}

pub(crate) fn run(repository_root: &Path, arguments: &[String]) -> Result<(), String> {
    let invocation = DuplicationInvocation::parse(arguments)?;
    let started_at = Instant::now();
    let mode_name = if invocation.should_write_baseline {
        "write"
    } else {
        "check"
    };
    println!(
        "[check-duplication] status=start mode={mode_name} against={}",
        invocation.baseline_ref
    );
    let (production_measurement, test_measurement) =
        measure_workspace(repository_root, &invocation)?;
    let measured_counts = RepeatedLineCounts {
        production_repeated_lines: production_measurement.repeated_line_count,
        test_repeated_lines: test_measurement.repeated_line_count,
    };
    judge_and_record(repository_root, &invocation, measured_counts)?;
    println!(
        "[check-duplication] status=success mode={mode_name} production_repeated_lines={} test_repeated_lines={} elapsed_seconds={:.1}",
        measured_counts.production_repeated_lines,
        measured_counts.test_repeated_lines,
        started_at.elapsed().as_secs_f64()
    );
    Ok(())
}

/// Reads both corpora, measures each and prints what was found.
fn measure_workspace(
    repository_root: &Path,
    invocation: &DuplicationInvocation,
) -> Result<(DuplicationMeasurement, DuplicationMeasurement), String> {
    let relative_paths = source_walk::collect_measured_source_paths(repository_root);
    println!(
        "[check-duplication] status=reading files={}",
        relative_paths.len()
    );
    let measured_corpora = source_walk::read_measured_corpora(repository_root, &relative_paths)?;
    println!("[check-duplication] status=measuring");
    let production_measurement = measure_repeated_blocks(&measured_corpora.production_files);
    let test_measurement = measure_repeated_blocks(&measured_corpora.test_files);
    for report_line in report::measurement_lines(
        "production",
        &production_measurement,
        invocation.top_pair_count,
    ) {
        println!("{report_line}");
    }
    for report_line in
        report::measurement_lines("test", &test_measurement, invocation.top_pair_count)
    {
        println!("{report_line}");
    }
    Ok((production_measurement, test_measurement))
}

/// Judges the counts against the committed baseline and, in write mode and
/// only on a pass, records them as the new baseline.
fn judge_and_record(
    repository_root: &Path,
    invocation: &DuplicationInvocation,
    measured_counts: RepeatedLineCounts,
) -> Result<(), String> {
    let committed_baseline_text =
        read_committed_baseline(repository_root, &invocation.baseline_ref)?;
    match judge_committed_baseline(committed_baseline_text.as_deref(), measured_counts)? {
        Some(corpus_verdicts) => {
            for corpus_verdict in &corpus_verdicts {
                println!(
                    "{}",
                    report::verdict_line(corpus_verdict, &invocation.baseline_ref)
                );
            }
            if has_grown_corpus(&corpus_verdicts) {
                return Err(
                    "[check-duplication] status=failed reason=repeated-lines-grew hint=\"each grown corpus above names its remedy\"".to_owned(),
                );
            }
        }
        None => println!(
            "[check-duplication] status=skipped-judgement reason=no-committed-baseline path={BASELINE_RELATIVE_PATH} against={}",
            invocation.baseline_ref
        ),
    }
    if invocation.should_write_baseline {
        write_baseline_atomically(repository_root, &render_baseline(measured_counts))?;
        println!("[check-duplication] status=written path={BASELINE_RELATIVE_PATH}");
    }
    Ok(())
}

/// The baseline committed at `baseline_ref`, or `None` when that ref has no
/// baseline file. An unknown ref fails the run.
fn read_committed_baseline(
    repository_root: &Path,
    baseline_ref: &str,
) -> Result<Option<String>, String> {
    let git_show_output = Command::new("git")
        .args(["show", &format!("{baseline_ref}:{BASELINE_RELATIVE_PATH}")])
        .current_dir(repository_root)
        .output()
        .map_err(|error| {
            format!("[check-duplication] status=failed reason=git-spawn-failed error={error}")
        })?;
    if git_show_output.status.success() {
        return String::from_utf8(git_show_output.stdout)
            .map(Some)
            .map_err(|error| {
                format!("[check-duplication] status=failed reason=non-utf8-baseline error={error}")
            });
    }
    let git_show_stderr = String::from_utf8_lossy(&git_show_output.stderr);
    if is_absent_at_ref_message(&git_show_stderr) {
        return Ok(None);
    }
    Err(format!(
        "[check-duplication] status=failed reason=no-committed-baseline against={baseline_ref} stderr={}",
        git_show_stderr.trim()
    ))
}

/// Writes beside the target and renames over it, so an interrupted run
/// leaves the old baseline intact rather than a truncated one.
fn write_baseline_atomically(repository_root: &Path, baseline_text: &str) -> Result<(), String> {
    let baseline_path = repository_root.join(BASELINE_RELATIVE_PATH);
    let staging_path = repository_root.join(format!("{BASELINE_RELATIVE_PATH}.tmp"));
    let write_outcome = std::fs::write(&staging_path, baseline_text)
        .and_then(|()| std::fs::rename(&staging_path, &baseline_path));
    if let Err(write_error) = write_outcome {
        let _ = std::fs::remove_file(&staging_path);
        return Err(format!(
            "[check-duplication] status=failed reason=write error={write_error}"
        ));
    }
    Ok(())
}
