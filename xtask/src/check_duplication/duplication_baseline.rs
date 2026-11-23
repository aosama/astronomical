//! The checked-in duplication baseline and the ratchet rule over it: the
//! repeated-line count of each corpus may stay where it is or go down, never
//! up.

use serde_json::{Value, json};

pub(crate) const BASELINE_RELATIVE_PATH: &str = "duplication-baseline.json";

const PRODUCTION_FIELD: &str = "production_repeated_lines";
const TEST_FIELD: &str = "test_repeated_lines";

/// The repeated-line count of each corpus: what the baseline holds and what a
/// run measures.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RepeatedLineCounts {
    pub(crate) production_repeated_lines: usize,
    pub(crate) test_repeated_lines: usize,
}

/// How one corpus compares with its baseline.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum CorpusComparison {
    Held,
    Lowered,
    Grown,
}

/// One corpus judged against the baseline.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct CorpusVerdict {
    pub(crate) corpus_name: &'static str,
    pub(crate) baseline_repeated_lines: usize,
    pub(crate) measured_repeated_lines: usize,
    pub(crate) comparison: CorpusComparison,
}

impl CorpusVerdict {
    fn judge(corpus_name: &'static str, baseline_lines: usize, measured_lines: usize) -> Self {
        let comparison = if measured_lines > baseline_lines {
            CorpusComparison::Grown
        } else if measured_lines < baseline_lines {
            CorpusComparison::Lowered
        } else {
            CorpusComparison::Held
        };
        Self {
            corpus_name,
            baseline_repeated_lines: baseline_lines,
            measured_repeated_lines: measured_lines,
            comparison,
        }
    }
}

/// Both corpora judged against the baseline: production first, then tests.
pub(crate) fn judge_against_baseline(
    baseline_counts: RepeatedLineCounts,
    measured_counts: RepeatedLineCounts,
) -> [CorpusVerdict; 2] {
    [
        CorpusVerdict::judge(
            "production",
            baseline_counts.production_repeated_lines,
            measured_counts.production_repeated_lines,
        ),
        CorpusVerdict::judge(
            "test",
            baseline_counts.test_repeated_lines,
            measured_counts.test_repeated_lines,
        ),
    ]
}

/// Judges the measured counts against the baseline committed at the ref the
/// run is checked against. `None` when that ref holds no baseline yet, which
/// is the case for exactly one commit: the one that adds the file.
pub(crate) fn judge_committed_baseline(
    committed_baseline_text: Option<&str>,
    measured_counts: RepeatedLineCounts,
) -> Result<Option<[CorpusVerdict; 2]>, String> {
    let Some(committed_baseline_text) = committed_baseline_text else {
        return Ok(None);
    };
    let baseline_counts = parse_baseline(committed_baseline_text)?;
    Ok(Some(judge_against_baseline(
        baseline_counts,
        measured_counts,
    )))
}

/// Whether git's refusal to show a file means the file is absent at that
/// ref, as opposed to the ref itself being unknown, which must fail the run
/// rather than skip the ratchet.
pub(crate) fn is_absent_at_ref_message(git_show_stderr: &str) -> bool {
    git_show_stderr.contains("does not exist in")
        || git_show_stderr.contains("exists on disk, but not in")
}

/// Whether any corpus grew past its baseline.
pub(crate) fn has_grown_corpus(corpus_verdicts: &[CorpusVerdict]) -> bool {
    for corpus_verdict in corpus_verdicts {
        if corpus_verdict.comparison == CorpusComparison::Grown {
            return true;
        }
    }
    false
}

/// Reads a baseline file. Both counts must be present as whole numbers;
/// anything else is refused, since a ratchet that guessed a missing count
/// would let that corpus grow unchecked.
pub(crate) fn parse_baseline(baseline_text: &str) -> Result<RepeatedLineCounts, String> {
    let baseline_document: Value = serde_json::from_str(baseline_text).map_err(|error| {
        format!("[check-duplication] status=failed reason=malformed-baseline error={error}")
    })?;
    Ok(RepeatedLineCounts {
        production_repeated_lines: read_count_field(&baseline_document, PRODUCTION_FIELD)?,
        test_repeated_lines: read_count_field(&baseline_document, TEST_FIELD)?,
    })
}

fn read_count_field(baseline_document: &Value, field_name: &str) -> Result<usize, String> {
    let Some(count) = baseline_document.get(field_name).and_then(Value::as_u64) else {
        return Err(format!(
            "[check-duplication] status=failed reason=baseline-field-missing-or-not-a-count field={field_name}"
        ));
    };
    usize::try_from(count).map_err(|error| {
        format!("[check-duplication] status=failed reason=baseline-count-out-of-range field={field_name} error={error}")
    })
}

/// The baseline file's text: pretty-printed with a trailing newline, so a
/// lowered count shows in review as a one-line change.
pub(crate) fn render_baseline(repeated_line_counts: RepeatedLineCounts) -> String {
    let baseline_document = json!({
        PRODUCTION_FIELD: repeated_line_counts.production_repeated_lines,
        TEST_FIELD: repeated_line_counts.test_repeated_lines,
    });
    // Pretty-printing a `Value` of two numbers cannot fail; the compact form
    // is only a fallback that still parses.
    let mut baseline_text = match serde_json::to_string_pretty(&baseline_document) {
        Ok(pretty_text) => pretty_text,
        Err(_) => baseline_document.to_string(),
    };
    baseline_text.push('\n');
    baseline_text
}
