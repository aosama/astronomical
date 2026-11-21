use std::fs;

use super::{SENT_REPORT_LOG_FILE_NAME, SentProblemReport, SentReportLog};

fn sent_report(sent_at_unix_seconds: u64, problem_line: &str) -> SentProblemReport {
    SentProblemReport {
        sent_at_unix_seconds,
        readable_lines: vec![
            "App: Astronomical 0.2.126".to_owned(),
            problem_line.to_owned(),
        ],
    }
}

#[test]
fn should_read_back_every_recorded_report_oldest_first() {
    let state_directory = tempfile::tempdir().expect("temporary folder should be created");
    let sent_report_log = SentReportLog::in_state_directory(state_directory.path());
    let first_report = sent_report(1_700_000_000, "Problem: model load failed");
    let second_report = sent_report(1_700_000_060, "Problem: engine did not start");
    sent_report_log
        .record(&first_report)
        .expect("first report should be recorded");
    sent_report_log
        .record(&second_report)
        .expect("second report should be recorded");
    assert_eq!(
        sent_report_log.sent_reports(),
        vec![first_report, second_report]
    );
}

#[test]
fn should_report_nothing_sent_when_the_log_does_not_exist() {
    let state_directory = tempfile::tempdir().expect("temporary folder should be created");
    let sent_report_log =
        SentReportLog::in_state_directory(&state_directory.path().join("missing"));
    assert!(sent_report_log.sent_reports().is_empty());
}

#[test]
fn should_skip_a_damaged_line_and_keep_the_others() {
    let state_directory = tempfile::tempdir().expect("temporary folder should be created");
    let sent_report_log = SentReportLog::in_state_directory(state_directory.path());
    let kept_report = sent_report(1_700_000_000, "Problem: model load failed");
    sent_report_log
        .record(&kept_report)
        .expect("report should be recorded");
    let log_file_path = state_directory.path().join(SENT_REPORT_LOG_FILE_NAME);
    let mut log_text = fs::read_to_string(&log_file_path).expect("log should exist");
    log_text.push_str("{\"sent_at_unix_seconds\": 17000\n");
    fs::write(&log_file_path, log_text).expect("damaged line should be written");
    assert_eq!(sent_report_log.sent_reports(), vec![kept_report]);
}
