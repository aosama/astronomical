use super::*;

fn judged_not_better_record() -> TrainingRecord {
    TrainingRecord {
        verdict: RecordedVerdict::JudgedNotBetter,
        labelled_question_count: 240,
        distinct_state_count: 48,
        warnings: vec!["right 61.0% of the time, down from 64.0%".to_owned()],
    }
}

#[test]
fn should_read_back_the_record_it_writes() {
    let training_record = judged_not_better_record();

    let read_back = parse_training_record(&training_record_document(&training_record)).unwrap();

    assert_eq!(read_back, training_record);
}

#[test]
fn should_write_the_verdict_by_its_wire_name() {
    let document_text =
        String::from_utf8(training_record_document(&judged_not_better_record())).unwrap();

    assert!(document_text.contains("\"verdict\": \"judged_not_better\""));
    assert!(document_text.ends_with('\n'));
}

#[test]
fn should_accept_a_record_of_one_question_about_one_text_with_no_warnings() {
    let training_record = TrainingRecord {
        verdict: RecordedVerdict::NotJudged,
        labelled_question_count: 1,
        distinct_state_count: 1,
        warnings: Vec::new(),
    };

    let read_back = parse_training_record(&training_record_document(&training_record)).unwrap();

    assert_eq!(read_back.verdict, RecordedVerdict::NotJudged);
    assert_eq!(read_back.labelled_question_count, 1);
}

#[test]
fn should_refuse_a_document_of_another_format() {
    let other_format = br#"{"format":"something-else","version":1,"verdict":"not_judged","labelled_question_count":1,"distinct_state_count":1,"warnings":[]}"#;

    assert!(matches!(
        parse_training_record(other_format),
        Err(TrainingRecordError::UnknownFormat { .. })
    ));
}

#[test]
fn should_refuse_a_document_that_is_not_json() {
    assert!(matches!(
        parse_training_record(b"not json"),
        Err(TrainingRecordError::Malformed(_))
    ));
}

#[test]
fn should_keep_an_earlier_worse_verdict_and_its_warnings_after_a_later_better_one() {
    let later = TrainingRecord {
        verdict: RecordedVerdict::JudgedBetter,
        labelled_question_count: 40,
        distinct_state_count: 40,
        warnings: Vec::new(),
    };

    let combined = later.after_earlier(Some(&judged_not_better_record()));

    assert_eq!(combined.verdict, RecordedVerdict::JudgedNotBetter);
    assert_eq!(combined.labelled_question_count, 40);
    assert_eq!(
        combined.warnings,
        vec!["earlier: right 61.0% of the time, down from 64.0%".to_owned()]
    );
}

#[test]
fn should_keep_the_later_record_alone_without_an_earlier_one() {
    assert_eq!(
        judged_not_better_record().after_earlier(None),
        judged_not_better_record()
    );
}

#[test]
fn should_rank_not_judged_above_judged_better() {
    let later = TrainingRecord {
        verdict: RecordedVerdict::JudgedBetter,
        labelled_question_count: 1,
        distinct_state_count: 1,
        warnings: Vec::new(),
    };
    let earlier = TrainingRecord {
        verdict: RecordedVerdict::NotJudged,
        ..later.clone()
    };

    assert_eq!(
        later.after_earlier(Some(&earlier)).verdict,
        RecordedVerdict::NotJudged
    );
}

#[test]
fn should_refuse_a_document_with_an_unknown_field() {
    let unknown_field = br#"{"format":"astronomical-decision-training-record","version":1,"verdict":"not_judged","labelled_question_count":1,"distinct_state_count":1,"warnings":[],"kept":true}"#;

    assert!(matches!(
        parse_training_record(unknown_field),
        Err(TrainingRecordError::Malformed(_))
    ));
}

#[test]
fn should_rank_judged_not_better_above_not_judged() {
    let later = TrainingRecord {
        verdict: RecordedVerdict::NotJudged,
        labelled_question_count: 1,
        distinct_state_count: 1,
        warnings: Vec::new(),
    };

    assert_eq!(
        later
            .after_earlier(Some(&judged_not_better_record()))
            .verdict,
        RecordedVerdict::JudgedNotBetter
    );
}

#[test]
fn should_mark_an_earlier_warning_once_and_keep_only_the_newest_warnings() {
    let mut earlier = judged_not_better_record();
    earlier.warnings = vec!["earlier: already marked".to_owned()];
    for warning_number in 0..MAXIMUM_RECORDED_WARNINGS {
        earlier.warnings.push(format!("warning {warning_number}"));
    }
    let later = TrainingRecord {
        warnings: vec!["the newest".to_owned()],
        ..judged_not_better_record()
    };

    let combined = later.after_earlier(Some(&earlier));

    assert_eq!(combined.warnings.len(), MAXIMUM_RECORDED_WARNINGS);
    assert_eq!(combined.warnings.last().unwrap(), "the newest");
    assert!(
        !combined
            .warnings
            .iter()
            .any(|warning| warning.starts_with("earlier: earlier: "))
    );
}
