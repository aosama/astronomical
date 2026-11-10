//! `cargo xtask crap-baseline` — the complexity ratchet over
//! `crap-baseline.json`, extended to functions the baseline does not list yet
//! (issue #1998).
//!
//! `cargo crap --fail-regression` only failed when a function already in the
//! baseline grew, so a new function could be as complex as it liked, and
//! nothing brought new functions under the ratchet when the baseline was
//! refreshed. This command keeps the growth rule and adds two more:
//!
//! - a function the committed baseline does not list may not exceed the
//!   cyclomatic cap in [`verdict`] unless `crap-admissions.json` names it;
//! - `--write` regenerates the baseline only after that check passes.
//!
//! Both modes judge the tree against a committed baseline, never the
//! working-tree copy: `HEAD`'s by default, as the pre-commit hook runs it, or
//! another ref's with `--against REF`, as the pre-push hook runs it against
//! `origin/main` so a branch built on an older baseline is judged by the
//! current one before it leaves this Mac. A plain commit therefore cannot
//! raise a baseline entry and grow its function together. What this gate
//! cannot see, and why:
//!
//! - growth split across two commits (a baseline-only commit made while the
//!   grown code is set aside), or folded into one afterwards with
//!   `git commit --amend`, a rebase or a squash; a commit made with
//!   `--no-verify`; or a hand edit merged without the hooks: requiring the
//!   committed baseline to equal a fresh report would also fail every merge
//!   that brings in `main`'s regenerated baseline;
//! - two branches that each pass on their own but grow a function once both
//!   are merged, when neither was pushed after the other landed: only a
//!   check of the merged tree sees that, which is why the pre-push hook
//!   judges against `origin/main`;
//! - a function renamed while it grew: cargo-crap reports it as new, so it is
//!   held only to the cap;
//! - same-named functions in one file whose start lines shift: cargo-crap
//!   then reports them as new and their baseline entries as removed, so they
//!   too are held only to the cap. Every removed function is printed, in
//!   both modes, so a reviewer can see either case.
//!
//! The gate scores the working tree, not the staged snapshot, as the
//! cargo-crap gate it replaces did.

pub(crate) mod admissions;
pub(crate) mod verdict;

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;

use serde_json::Value;

use admissions::{ADMISSIONS_RELATIVE_PATH, parse_admissions};
use verdict::{
    CrapBaselineVerdict, MAXIMUM_NEW_FUNCTION_CYCLOMATIC_COMPLEXITY, judge_delta_report,
};

pub(crate) const BASELINE_RELATIVE_PATH: &str = "crap-baseline.json";
/// Where the committed baseline is copied for cargo-crap to read; inside
/// `target/` so it is never staged.
const COMMITTED_BASELINE_COPY_RELATIVE_PATH: &str = "target/crap-baseline/committed-baseline.json";
const BASELINE_REPORT_SCHEMA_SUFFIX: &str = "/schemas/report-v1.json";
const MISSING_TOOL_HINT: &str =
    "cargo-crap is required: brew install cargo-binstall && cargo binstall cargo-crap";

const DEFAULT_BASELINE_REF: &str = "HEAD";

/// The parsed command line of `cargo xtask crap-baseline`.
#[derive(Debug, PartialEq)]
pub(crate) struct CrapBaselineInvocation {
    pub(crate) should_write_baseline: bool,
    /// The git ref whose committed baseline the tree is judged against.
    pub(crate) baseline_ref: String,
}

impl Default for CrapBaselineInvocation {
    fn default() -> Self {
        Self {
            should_write_baseline: false,
            baseline_ref: DEFAULT_BASELINE_REF.to_owned(),
        }
    }
}

impl CrapBaselineInvocation {
    pub(crate) fn parse(arguments: &[String]) -> Result<Self, String> {
        let mut invocation = Self::default();
        let mut remaining_arguments = arguments.iter();
        while let Some(argument) = remaining_arguments.next() {
            invocation.apply_argument(argument, &mut remaining_arguments)?;
        }
        Ok(invocation)
    }

    /// Applies one argument, taking `--against`'s value from the arguments
    /// that follow it. A value starting with `-` is another flag, not a ref.
    fn apply_argument<'argument>(
        &mut self,
        argument: &str,
        remaining_arguments: &mut impl Iterator<Item = &'argument String>,
    ) -> Result<(), String> {
        match argument {
            "--write" => self.should_write_baseline = true,
            "--against" => {
                self.baseline_ref = remaining_arguments
                    .next()
                    .filter(|baseline_ref| !baseline_ref.starts_with('-'))
                    .ok_or(
                        "[crap-baseline] status=failed reason=missing-value-for argument=--against",
                    )?
                    .clone();
            }
            unknown_argument => {
                return Err(format!(
                    "[crap-baseline] status=failed reason=unknown-argument argument={unknown_argument} hint=\"the arguments are --write and --against REF\""
                ));
            }
        }
        Ok(())
    }
}

/// The two cargo-crap scans the ratchet needs, behind a trait so the rule
/// that nothing is written unless the check passes can be tested without
/// running cargo-crap or git.
pub(crate) trait CrapReportSource {
    /// The delta report of the current tree against the baseline committed
    /// at `baseline_ref`.
    fn delta_against_committed_baseline(&self, baseline_ref: &str) -> Result<String, String>;
    /// A fresh `report-v1` report of the current tree.
    fn fresh_baseline_report(&self) -> Result<String, String>;
}

pub(crate) fn run(repository_root: &Path, arguments: &[String]) -> Result<(), String> {
    let invocation = CrapBaselineInvocation::parse(arguments)?;
    let admissions_text = read_admissions_text(repository_root)?;
    let report_source = CargoCrapReportSource {
        repository_root: repository_root.to_path_buf(),
    };
    run_with_source(
        repository_root,
        &invocation,
        &admissions_text,
        &report_source,
    )
}

/// Judges the tree and, in write mode and only on a pass, replaces the
/// baseline with the fresh report. The fresh report is taken first so the
/// bytes written come from the same tree the verdict then judges.
pub(crate) fn run_with_source(
    repository_root: &Path,
    invocation: &CrapBaselineInvocation,
    admissions_text: &str,
    report_source: &dyn CrapReportSource,
) -> Result<(), String> {
    let started_at = Instant::now();
    let mode_name = if invocation.should_write_baseline {
        "write"
    } else {
        "check"
    };
    println!(
        "[crap-baseline] status=start mode={mode_name} against={}",
        invocation.baseline_ref
    );
    let admitted_function_labels = parse_admissions(admissions_text)?;
    let fresh_baseline_text = take_validated_fresh_baseline(invocation, report_source)?;
    println!("[crap-baseline] status=scoring step=delta-against-committed-baseline");
    let delta_report_text =
        report_source.delta_against_committed_baseline(&invocation.baseline_ref)?;
    let verdict = judge_delta_report(&delta_report_text, &admitted_function_labels)?;
    print_verdict(&verdict);
    if !verdict.passes() {
        return Err(verdict.failure_summary());
    }
    if let Some(fresh_baseline_text) = fresh_baseline_text {
        write_baseline_atomically(repository_root, &fresh_baseline_text)?;
        println!("[crap-baseline] status=written path={BASELINE_RELATIVE_PATH}");
    }
    println!(
        "[crap-baseline] status=success mode={mode_name} new_functions={} removed_functions={} elapsed_seconds={:.1}",
        verdict.new_function_count,
        verdict.removed_functions.len(),
        started_at.elapsed().as_secs_f64()
    );
    Ok(())
}

/// In write mode, the fresh report the baseline would become, checked to be a
/// baseline report before anything is judged; `None` in check mode, which
/// never pays for the extra scan.
fn take_validated_fresh_baseline(
    invocation: &CrapBaselineInvocation,
    report_source: &dyn CrapReportSource,
) -> Result<Option<String>, String> {
    if !invocation.should_write_baseline {
        return Ok(None);
    }
    println!("[crap-baseline] status=scoring step=fresh-baseline");
    let fresh_baseline_text = report_source.fresh_baseline_report()?;
    validate_fresh_baseline_report(&fresh_baseline_text)?;
    Ok(Some(fresh_baseline_text))
}

fn read_admissions_text(repository_root: &Path) -> Result<String, String> {
    std::fs::read_to_string(repository_root.join(ADMISSIONS_RELATIVE_PATH)).map_err(|error| {
        format!(
            "[crap-baseline] status=failed reason=admissions-unreadable path={ADMISSIONS_RELATIVE_PATH} error={error}"
        )
    })
}

/// Each violation kind gets its own hint: an admission can rescue a new
/// function above the cap, never a baseline function that grew.
fn print_verdict(verdict: &CrapBaselineVerdict) {
    for grown_function in &verdict.grown_functions {
        eprintln!(
            "[crap-baseline] violation=grown function={grown_function} hint=\"bring it back to its baseline score; an admission cannot cover growth\""
        );
    }
    for over_cap_function in &verdict.over_cap_new_functions {
        eprintln!(
            "[crap-baseline] violation=new-function-over-cap limit={MAXIMUM_NEW_FUNCTION_CYCLOMATIC_COMPLEXITY} function={over_cap_function} hint=\"split it, or admit it with a reason in {ADMISSIONS_RELATIVE_PATH}\""
        );
    }
    for stale_admission in &verdict.stale_admissions {
        eprintln!(
            "[crap-baseline] violation=stale-admission function={stale_admission} hint=\"remove it from {ADMISSIONS_RELATIVE_PATH}\""
        );
    }
    for ambiguous_admission in &verdict.ambiguous_admissions {
        eprintln!(
            "[crap-baseline] violation=ambiguous-admission function={ambiguous_admission} hint=\"several same-named functions share this file; rename the one to admit\""
        );
    }
    for admitted_function in &verdict.admitted_functions {
        println!("[crap-baseline] admitted function={admitted_function}");
    }
    for removed_function in &verdict.removed_functions {
        println!("[crap-baseline] removed function={removed_function}");
    }
}

/// The baseline is written byte for byte as cargo-crap produced it, since
/// re-serializing could reorder keys or reformat floats in a file cargo-crap
/// reads back, so the text is only checked here, never rebuilt.
fn validate_fresh_baseline_report(fresh_baseline_text: &str) -> Result<(), String> {
    let fresh_baseline: Value = serde_json::from_str(fresh_baseline_text).map_err(|error| {
        format!("[crap-baseline] status=failed reason=malformed-fresh-baseline error={error}")
    })?;
    let has_report_schema = fresh_baseline
        .get("$schema")
        .and_then(Value::as_str)
        .is_some_and(|schema_url| schema_url.ends_with(BASELINE_REPORT_SCHEMA_SUFFIX));
    let has_entries = fresh_baseline.get("entries").is_some_and(Value::is_array);
    if has_report_schema && has_entries {
        Ok(())
    } else {
        Err("[crap-baseline] status=failed reason=fresh-baseline-is-not-a-report".to_owned())
    }
}

/// Writes beside the target and renames over it, so an interrupted run
/// leaves the old baseline intact rather than a truncated one.
fn write_baseline_atomically(
    repository_root: &Path,
    fresh_baseline_text: &str,
) -> Result<(), String> {
    let baseline_path = repository_root.join(BASELINE_RELATIVE_PATH);
    let staging_path = repository_root.join(format!("{BASELINE_RELATIVE_PATH}.tmp"));
    std::fs::write(&staging_path, fresh_baseline_text)
        .and_then(|()| std::fs::rename(&staging_path, &baseline_path))
        .map_err(|write_error| discard_staging_file(&staging_path, &write_error))
}

/// The staging file is ignored by git, but a stale copy would sit beside the
/// baseline until someone noticed it, so a failed write removes it.
fn discard_staging_file(staging_path: &Path, write_error: &std::io::Error) -> String {
    let _ = std::fs::remove_file(staging_path);
    format!("[crap-baseline] status=failed reason=write error={write_error}")
}

/// The real scans: git for the committed baseline, cargo-crap for scoring.
struct CargoCrapReportSource {
    repository_root: PathBuf,
}

impl CrapReportSource for CargoCrapReportSource {
    fn delta_against_committed_baseline(&self, baseline_ref: &str) -> Result<String, String> {
        let committed_baseline_copy = self.copy_committed_baseline(baseline_ref)?;
        let committed_baseline_argument = committed_baseline_copy.to_string_lossy();
        self.run_cargo_crap(&["--baseline", committed_baseline_argument.as_ref()])
    }

    fn fresh_baseline_report(&self) -> Result<String, String> {
        self.run_cargo_crap(&[])
    }
}

impl CargoCrapReportSource {
    fn copy_committed_baseline(&self, baseline_ref: &str) -> Result<PathBuf, String> {
        let git_show_output = Command::new("git")
            .args(["show", &format!("{baseline_ref}:{BASELINE_RELATIVE_PATH}")])
            .current_dir(&self.repository_root)
            .output()
            .map_err(|error| {
                format!("[crap-baseline] status=failed reason=git-spawn-failed error={error}")
            })?;
        if !git_show_output.status.success() {
            return Err(format!(
                "[crap-baseline] status=failed reason=no-committed-baseline against={baseline_ref} stderr={}",
                String::from_utf8_lossy(&git_show_output.stderr).trim()
            ));
        }
        let committed_baseline_copy = self
            .repository_root
            .join(COMMITTED_BASELINE_COPY_RELATIVE_PATH);
        if let Some(copy_directory) = committed_baseline_copy.parent() {
            std::fs::create_dir_all(copy_directory).map_err(|error| {
                format!("[crap-baseline] status=failed reason=copy-directory error={error}")
            })?;
        }
        std::fs::write(&committed_baseline_copy, git_show_output.stdout).map_err(|error| {
            format!("[crap-baseline] status=failed reason=committed-baseline-copy error={error}")
        })?;
        Ok(committed_baseline_copy)
    }

    /// Runs `cargo crap --workspace --format json` plus `extra_arguments` from
    /// the repository root, where `.cargo-crap.toml` lives. Its warnings pass
    /// straight through, since a skipped file would otherwise vanish silently
    /// into the removed list.
    fn run_cargo_crap(&self, extra_arguments: &[&str]) -> Result<String, String> {
        let cargo_crap_output = Command::new("cargo")
            .args(["crap", "--workspace", "--format", "json"])
            .args(extra_arguments)
            .current_dir(&self.repository_root)
            .stderr(Stdio::inherit())
            .output()
            .map_err(|error| {
                format!(
                    "[crap-baseline] status=failed reason=spawn-failed hint=\"{MISSING_TOOL_HINT}\" error={error}"
                )
            })?;
        if !cargo_crap_output.status.success() {
            return Err(format!(
                "[crap-baseline] status=failed reason=cargo-crap-exited exit_status={} hint=\"{MISSING_TOOL_HINT}\"",
                cargo_crap_output.status
            ));
        }
        String::from_utf8(cargo_crap_output.stdout).map_err(|error| {
            format!("[crap-baseline] status=failed reason=non-utf8-report error={error}")
        })
    }
}
