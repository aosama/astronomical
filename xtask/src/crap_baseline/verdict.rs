//! What a cargo-crap delta report means for the complexity ratchet.
//!
//! Matching a function to its baseline entry stays cargo-crap's job: its
//! delta report labels each function `unchanged`, `improved`, `regressed`,
//! `new` or `moved` (a pure relocation with an unchanged score). This module
//! reads those labels and nothing else, and fails closed on anything it does
//! not recognise, because a changed report format must never read as a pass.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::Value;

/// Highest cyclomatic complexity a function outside the baseline may have.
/// Fifteen was the 99th percentile of every function on `main` when this
/// gate was introduced (2026-11-09): new code may be as branchy as the most
/// complex one percent of the code that existed then, and no more. The
/// 100-line clippy budget bounds length; this bounds branching.
pub(crate) const MAXIMUM_NEW_FUNCTION_CYCLOMATIC_COMPLEXITY: f64 = 15.0;

const KNOWN_ENTRY_STATUSES: &[&str] = &["unchanged", "improved", "regressed", "new", "moved"];

/// The ratchet's verdict on one delta report.
#[derive(Debug, Default, PartialEq)]
pub(crate) struct CrapBaselineVerdict {
    /// Baseline functions whose score grew: `label baseline_crap=.. crap=..`.
    pub(crate) grown_functions: Vec<String>,
    /// New functions above the cap that no admission covers.
    pub(crate) over_cap_new_functions: Vec<String>,
    /// New functions above the cap that an admission covers.
    pub(crate) admitted_functions: Vec<String>,
    /// Admissions naming no function in the tree any more.
    pub(crate) stale_admissions: Vec<String>,
    /// Admissions naming several same-named functions in one file; one
    /// label must never cover more than the function it was written for.
    pub(crate) ambiguous_admissions: Vec<String>,
    pub(crate) new_function_count: usize,
    pub(crate) removed_functions: Vec<String>,
}

impl CrapBaselineVerdict {
    /// The closing line of a failed run, one count per violation kind.
    pub(crate) fn failure_summary(&self) -> String {
        format!(
            "[crap-baseline] status=failed grown={} over_cap_new={} stale_admissions={} ambiguous_admissions={}",
            self.grown_functions.len(),
            self.over_cap_new_functions.len(),
            self.stale_admissions.len(),
            self.ambiguous_admissions.len()
        )
    }

    pub(crate) fn passes(&self) -> bool {
        self.grown_functions.is_empty()
            && self.over_cap_new_functions.is_empty()
            && self.stale_admissions.is_empty()
            && self.ambiguous_admissions.is_empty()
    }
}

/// Judges a `cargo crap --baseline … --format json` delta report against
/// the admitted `file:function` labels.
pub(crate) fn judge_delta_report(
    delta_report_text: &str,
    admitted_function_labels: &BTreeSet<String>,
) -> Result<CrapBaselineVerdict, String> {
    let delta_report: Value = serde_json::from_str(delta_report_text).map_err(|error| {
        format!("[crap-baseline] status=failed reason=malformed-report error={error}")
    })?;
    let report_entries = required_array(&delta_report, "entries")?;
    let mut verdict = CrapBaselineVerdict {
        removed_functions: removed_function_labels(required_array(&delta_report, "removed")?)?,
        ..CrapBaselineVerdict::default()
    };
    let mut label_to_entry_count: BTreeMap<String, usize> = BTreeMap::new();
    for report_entry in report_entries {
        let scored_function = ScoredFunction::from_report_entry(report_entry)?;
        *label_to_entry_count
            .entry(scored_function.label.clone())
            .or_default() += 1;
        judge_scored_function(&mut verdict, scored_function, admitted_function_labels);
    }
    for admitted_label in admitted_function_labels {
        match label_to_entry_count.get(admitted_label) {
            None => verdict.stale_admissions.push(admitted_label.clone()),
            Some(&entry_count) if entry_count > 1 => {
                verdict.ambiguous_admissions.push(admitted_label.clone());
            }
            Some(_) => {}
        }
    }
    Ok(verdict)
}

/// An admission only ever covers a new function above the cap: a function
/// already in the baseline that grew is refused whatever is admitted.
fn judge_scored_function(
    verdict: &mut CrapBaselineVerdict,
    scored_function: ScoredFunction<'_>,
    admitted_function_labels: &BTreeSet<String>,
) {
    match scored_function.status {
        "regressed" => verdict.grown_functions.push(format!(
            "{} baseline_crap={} crap={}",
            scored_function.label,
            scored_function.baseline_crap.unwrap_or_default(),
            scored_function.crap
        )),
        "new" => {
            verdict.new_function_count += 1;
            if scored_function.cyclomatic <= MAXIMUM_NEW_FUNCTION_CYCLOMATIC_COMPLEXITY {
                return;
            }
            let described_function = format!(
                "{} cyclomatic={}",
                scored_function.label, scored_function.cyclomatic
            );
            if admitted_function_labels.contains(&scored_function.label) {
                verdict.admitted_functions.push(described_function);
            } else {
                verdict.over_cap_new_functions.push(described_function);
            }
        }
        _ => {}
    }
}

/// The fields of one delta entry the verdict reads.
struct ScoredFunction<'report> {
    label: String,
    status: &'report str,
    cyclomatic: f64,
    crap: f64,
    /// Absent only for a new function, which has no baseline score.
    baseline_crap: Option<f64>,
}

impl<'report> ScoredFunction<'report> {
    fn from_report_entry(report_entry: &'report Value) -> Result<Self, String> {
        let status = entry_text(report_entry, "status")?;
        if !KNOWN_ENTRY_STATUSES.contains(&status) {
            return Err(format!(
                "[crap-baseline] status=failed reason=unknown-entry-status entry_status={status}"
            ));
        }
        let baseline_crap = if status == "new" {
            None
        } else {
            Some(entry_number(report_entry, "baseline_crap")?)
        };
        Ok(Self {
            label: function_label(report_entry)?,
            status,
            cyclomatic: entry_number(report_entry, "cyclomatic")?,
            crap: entry_number(report_entry, "crap")?,
            baseline_crap,
        })
    }
}

fn removed_function_labels(removed_entries: &[Value]) -> Result<Vec<String>, String> {
    let mut removed_labels = Vec::with_capacity(removed_entries.len());
    for removed_entry in removed_entries {
        removed_labels.push(format!(
            "{} baseline_crap={}",
            function_label(removed_entry)?,
            entry_number(removed_entry, "baseline_crap")?
        ));
    }
    Ok(removed_labels)
}

/// `file:function`, the form admissions and violation lines both use.
fn function_label(report_entry: &Value) -> Result<String, String> {
    Ok(format!(
        "{}:{}",
        entry_text(report_entry, "file")?,
        entry_text(report_entry, "function")?
    ))
}

fn required_array<'report>(
    delta_report: &'report Value,
    field_name: &str,
) -> Result<&'report [Value], String> {
    delta_report
        .get(field_name)
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .ok_or_else(|| {
            format!("[crap-baseline] status=failed reason=report-without-field field={field_name}")
        })
}

fn entry_text<'report>(
    report_entry: &'report Value,
    field_name: &str,
) -> Result<&'report str, String> {
    report_entry
        .get(field_name)
        .and_then(Value::as_str)
        .ok_or_else(|| missing_field_error(field_name))
}

fn entry_number(report_entry: &Value, field_name: &str) -> Result<f64, String> {
    report_entry
        .get(field_name)
        .and_then(Value::as_f64)
        .ok_or_else(|| missing_field_error(field_name))
}

fn missing_field_error(field_name: &str) -> String {
    format!("[crap-baseline] status=failed reason=entry-without-field field={field_name}")
}
