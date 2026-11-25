use super::{
    DrafterInUse, FigureSpread, MeasuringMachine, ReleaseSpeedLedger, SpeedMeasurement,
    TaskFigures, WeightsPlacement, merged_ledger, newest_ledger, read_ledger, version_parts,
    write_ledger,
};

pub(crate) fn spread(median: f64, lowest: f64, highest: f64) -> FigureSpread {
    FigureSpread {
        median,
        lowest,
        highest,
    }
}

pub(crate) fn measurement(model_id: &str, task: &str, reply_median: f64) -> SpeedMeasurement {
    SpeedMeasurement {
        model_id: model_id.to_owned(),
        model_revision: Some("revision-a".to_owned()),
        maximum_mlx_memory_gb: Some(16),
        mlx_memory_limit_bytes: 16 * 1_073_741_824,
        weights_placement: WeightsPlacement::Streamed,
        drafter: DrafterInUse::Dflash,
        multi_token_prediction_draft_tokens: Some(1),
        task: task.to_owned(),
        temperature: 0.7,
        reply_tokens_requested: 500,
        figures: TaskFigures {
            round_count: 3,
            prompt_tokens: 10_120,
            prefill_tokens_per_second: spread(2_400.0, 2_350.0, 2_450.0),
            seconds_to_first_token: spread(4.2, 4.1, 4.3),
            reply_tokens_per_second: spread(reply_median, reply_median - 2.0, reply_median + 2.0),
        },
    }
}

pub(crate) fn ledger(version: &str, measurements: Vec<SpeedMeasurement>) -> ReleaseSpeedLedger {
    ReleaseSpeedLedger {
        astronomical_version: version.to_owned(),
        source_commit: format!("commit-of-{version}"),
        measured_on_date: "2026-11-26".to_owned(),
        machine: MeasuringMachine {
            chip: "M5 Pro".to_owned(),
            memory_bytes: 48 * 1_073_741_824,
        },
        measurements,
    }
}

#[test]
fn should_keep_the_median_lowest_and_highest_of_rounds_to_two_decimals() {
    assert_eq!(
        FigureSpread::of_rounds(&[61.234, 70.555, 65.0]),
        Some(spread(65.0, 61.23, 70.56))
    );
    assert_eq!(
        FigureSpread::of_rounds(&[60.0, 70.0, 71.0, 90.0]),
        Some(spread(70.5, 60.0, 90.0)),
        "an even number of rounds takes the midpoint of the middle two"
    );
}

#[test]
fn should_have_no_spread_without_rounds() {
    assert_eq!(FigureSpread::of_rounds(&[]), None);
}

#[test]
fn should_call_spreads_apart_only_when_they_do_not_overlap() {
    let earlier = spread(55.0, 53.0, 57.0);

    assert!(spread(65.0, 61.0, 69.0).is_apart_from(&earlier));
    assert!(spread(45.0, 43.0, 47.0).is_apart_from(&earlier));
    assert!(
        !spread(58.0, 57.0, 60.0).is_apart_from(&earlier),
        "touching ranges are not apart"
    );
    assert!(!spread(55.5, 52.0, 58.0).is_apart_from(&earlier));
}

#[test]
fn should_replace_a_remeasured_workload_and_keep_the_others() {
    let earlier_sitting = ledger(
        "0.2.135",
        vec![
            measurement("Model-A", "code_long_prompt", 50.0),
            measurement("Model-B", "code_long_prompt", 20.0),
        ],
    );
    let later_sitting = ledger(
        "0.2.135",
        vec![measurement("Model-A", "code_long_prompt", 60.0)],
    );

    let merged = merged_ledger(Some(earlier_sitting), later_sitting).expect("one build, one Mac");

    let reply_medians: Vec<(String, f64)> = merged
        .measurements
        .iter()
        .map(|kept| {
            (
                kept.model_id.clone(),
                kept.figures.reply_tokens_per_second.median,
            )
        })
        .collect();
    assert_eq!(
        reply_medians,
        vec![("Model-A".to_owned(), 60.0), ("Model-B".to_owned(), 20.0)]
    );
}

#[test]
fn should_refuse_to_merge_sittings_from_another_build_or_mac() {
    let earlier_sitting = ledger(
        "0.2.135",
        vec![measurement("Model-A", "code_long_prompt", 50.0)],
    );
    let mut other_build = ledger("0.2.135", Vec::new());
    other_build.source_commit = "another-commit".to_owned();
    let mut other_mac = ledger("0.2.135", Vec::new());
    other_mac.machine.chip = "M4".to_owned();

    assert!(merged_ledger(Some(earlier_sitting.clone()), other_build).is_err());
    assert!(merged_ledger(Some(earlier_sitting), other_mac).is_err());
}

#[test]
fn should_order_versions_by_number_and_reject_names_that_are_not_versions() {
    assert_eq!(version_parts("0.2.134"), Some(vec![0, 2, 134]));
    assert!(version_parts("0.2.140") > version_parts("0.2.99"));
    assert_eq!(version_parts("reference_set"), None);
    assert_eq!(version_parts("0.2.x"), None);
}

#[test]
fn should_write_read_and_find_the_newest_ledger() {
    let repository_root = tempfile::tempdir().expect("a temporary repository");
    assert_eq!(
        newest_ledger(repository_root.path()).expect("no ledger directory yet"),
        None
    );

    for version in ["0.2.99", "0.2.140", "0.2.134"] {
        write_ledger(
            repository_root.path(),
            ledger(
                version,
                vec![measurement("Model-A", "essay_long_prompt", 30.0)],
            ),
        )
        .expect("a written ledger");
    }
    let written_file = repository_root
        .path()
        .join("registry/release_speed/0.2.134.json");

    assert_eq!(
        read_ledger(&written_file)
            .expect("a readable ledger")
            .astronomical_version,
        "0.2.134"
    );
    assert_eq!(
        newest_ledger(repository_root.path())
            .expect("readable ledgers")
            .map(|newest| newest.astronomical_version),
        Some("0.2.140".to_owned())
    );
}

#[test]
fn should_refuse_a_file_with_unknown_fields() {
    let repository_root = tempfile::tempdir().expect("a temporary repository");
    let ledger_file = repository_root.path().join("0.2.1.json");
    std::fs::write(
        &ledger_file,
        r#"{"astronomical_version": "0.2.1", "guess": 1}"#,
    )
    .expect("a written file");

    assert!(read_ledger(&ledger_file).is_err());
}
