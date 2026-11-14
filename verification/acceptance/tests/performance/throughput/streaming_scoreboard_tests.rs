use std::fs;

use super::{
    ExpertStorageReads, entitlement_share_filled, limit_overflow_count_in,
    limit_overflow_count_in_worker_logs, reads_since,
};

#[test]
fn should_report_the_filled_share_of_the_expert_entitlement() {
    assert_eq!(entitlement_share_filled(9_000, 1_000), Some(0.9));
}

#[test]
fn should_report_a_full_share_when_nothing_is_left_unfilled() {
    assert_eq!(entitlement_share_filled(4_096, 0), Some(1.0));
}

#[test]
fn should_report_an_empty_share_when_nothing_is_kept() {
    assert_eq!(entitlement_share_filled(0, 4_096), Some(0.0));
}

#[test]
fn should_report_no_share_for_a_model_without_an_expert_entitlement() {
    assert_eq!(entitlement_share_filled(0, 0), None);
}

#[test]
fn should_count_every_limit_overflow_line_in_a_worker_log() {
    let worker_log_text = "INFO executing prompt-prefill chunk\n\
        WARN prefill chunk forward exceeded the MLX active-memory limit request_id=2\n\
        WARN recovered prefill memory exhaustion\n\
        WARN prefill chunk forward exceeded the MLX active-memory limit request_id=3\n";
    assert_eq!(limit_overflow_count_in(worker_log_text), 2);
}

#[test]
fn should_count_no_overflow_in_a_clean_or_empty_worker_log() {
    assert_eq!(
        limit_overflow_count_in("INFO executing prompt-prefill chunk\n"),
        0
    );
    assert_eq!(limit_overflow_count_in(""), 0);
}

#[test]
fn should_report_what_was_read_after_the_earlier_snapshot() {
    let reads_after_warmup = ExpertStorageReads {
        prompt_read_bytes: 1_000,
        reply_read_bytes: 200,
    };
    let reads_at_end = ExpertStorageReads {
        prompt_read_bytes: 41_000,
        reply_read_bytes: 900,
    };
    assert_eq!(
        reads_since(reads_after_warmup, reads_at_end),
        ExpertStorageReads {
            prompt_read_bytes: 40_000,
            reply_read_bytes: 700,
        }
    );
}

#[test]
fn should_report_no_reads_when_the_counters_did_not_move() {
    let session_reads = ExpertStorageReads {
        prompt_read_bytes: 5,
        reply_read_bytes: 7,
    };
    assert_eq!(
        reads_since(session_reads, session_reads),
        ExpertStorageReads {
            prompt_read_bytes: 0,
            reply_read_bytes: 0,
        }
    );
}

#[test]
fn should_report_zero_instead_of_wrapping_when_the_later_counters_are_smaller() {
    let reads_before_reload = ExpertStorageReads {
        prompt_read_bytes: 9_000,
        reply_read_bytes: 9_000,
    };
    let reads_after_reload = ExpertStorageReads {
        prompt_read_bytes: 10,
        reply_read_bytes: 20,
    };
    assert_eq!(
        reads_since(reads_before_reload, reads_after_reload),
        ExpertStorageReads {
            prompt_read_bytes: 0,
            reply_read_bytes: 0,
        }
    );
}

#[test]
fn should_sum_overflows_across_worker_logs_and_ignore_other_files() {
    let logging_directory = tempfile::tempdir().expect("a logging directory should be created");
    let overflow_line = "WARN prefill chunk forward exceeded the MLX active-memory limit\n";
    fs::write(
        logging_directory.path().join("worker.2026-11-14-10.log"),
        overflow_line,
    )
    .expect("the first worker log should be written");
    fs::write(
        logging_directory.path().join("worker.2026-11-14-11.log"),
        format!("{overflow_line}INFO executing prompt-prefill chunk\n{overflow_line}"),
    )
    .expect("the second worker log should be written");
    fs::write(
        logging_directory.path().join("performance.jsonl"),
        overflow_line,
    )
    .expect("the unrelated file should be written");
    assert_eq!(
        limit_overflow_count_in_worker_logs(logging_directory.path()),
        Some(3)
    );
}

#[test]
fn should_report_zero_overflows_when_the_only_worker_log_is_clean() {
    let logging_directory = tempfile::tempdir().expect("a logging directory should be created");
    fs::write(
        logging_directory.path().join("worker.2026-11-14-10.log"),
        "INFO executing prompt-prefill chunk\n",
    )
    .expect("the worker log should be written");
    assert_eq!(
        limit_overflow_count_in_worker_logs(logging_directory.path()),
        Some(0)
    );
}

#[test]
fn should_report_no_count_when_there_is_no_worker_log_to_read() {
    let logging_directory = tempfile::tempdir().expect("a logging directory should be created");
    assert_eq!(
        limit_overflow_count_in_worker_logs(logging_directory.path()),
        None
    );
    assert_eq!(
        limit_overflow_count_in_worker_logs(&logging_directory.path().join("missing")),
        None
    );
}
