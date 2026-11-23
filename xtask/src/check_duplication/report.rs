//! The lines `cargo xtask check-duplication` prints, built as strings so the
//! wording is tested without capturing standard output.

use super::duplication_baseline::{CorpusComparison, CorpusVerdict};
use super::repeated_blocks::{DuplicationMeasurement, RepeatedLinePair};

/// The totals of one corpus, then its largest area pairs and file pairs, at
/// most `top_pair_count` of each, so a reader sees where the duplication is.
pub(crate) fn measurement_lines(
    corpus_name: &str,
    measurement: &DuplicationMeasurement,
    top_pair_count: usize,
) -> Vec<String> {
    let mut report_lines = vec![format!(
        "[check-duplication] corpus={corpus_name} files={} code_lines={} repeated_lines={} repeated_percentage={:.1}",
        measurement.file_count,
        measurement.code_line_count,
        measurement.repeated_line_count,
        measurement.repeated_percentage()
    )];
    append_pair_lines(
        &mut report_lines,
        &format!(
            "[check-duplication] corpus={corpus_name} top_area_pairs (lines in the first area that repeat in the second):"
        ),
        &measurement.area_pairs,
        top_pair_count,
    );
    append_pair_lines(
        &mut report_lines,
        &format!(
            "[check-duplication] corpus={corpus_name} top_file_pairs (lines in the first file that repeat in the second):"
        ),
        &measurement.file_pairs,
        top_pair_count,
    );
    report_lines
}

fn append_pair_lines(
    report_lines: &mut Vec<String>,
    heading_line: &str,
    repeated_line_pairs: &[RepeatedLinePair],
    top_pair_count: usize,
) {
    if top_pair_count == 0 || repeated_line_pairs.is_empty() {
        return;
    }
    report_lines.push(heading_line.to_owned());
    for repeated_line_pair in repeated_line_pairs.iter().take(top_pair_count) {
        report_lines.push(format!(
            "  {:>6}  {}  <->  {}",
            repeated_line_pair.repeated_line_count,
            repeated_line_pair.first_place,
            repeated_line_pair.second_place
        ));
    }
}

/// One line per corpus saying how it compares with the committed baseline,
/// with the remedy for the case that needs one.
pub(crate) fn verdict_line(corpus_verdict: &CorpusVerdict, baseline_ref: &str) -> String {
    let (verdict_status, hint) = match corpus_verdict.comparison {
        CorpusComparison::Held => ("held", String::new()),
        CorpusComparison::Lowered => (
            "lowered",
            " hint=\"run cargo xtask check-duplication --write and commit the baseline to lock in the lower count\"".to_owned(),
        ),
        CorpusComparison::Grown => (
            "grown",
            " hint=\"move the copied block into one shared place, using the pairs above to find it; the baseline may only go down\"".to_owned(),
        ),
    };
    format!(
        "[check-duplication] verdict={verdict_status} corpus={} baseline_repeated_lines={} measured_repeated_lines={} against={baseline_ref}{hint}",
        corpus_verdict.corpus_name,
        corpus_verdict.baseline_repeated_lines,
        corpus_verdict.measured_repeated_lines
    )
}
