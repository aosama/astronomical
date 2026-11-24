use std::collections::HashMap;
use std::time::Duration;

use crate::check_duplication::DuplicationInvocation;
use crate::check_duplication::duplication_baseline::{
    CorpusComparison, RepeatedLineCounts, has_grown_corpus, is_absent_at_ref_message,
    judge_against_baseline, judge_committed_baseline, parse_baseline, render_baseline,
};
use crate::check_duplication::progress::{is_progress_report_due, progress_line};
use crate::check_duplication::repeated_blocks::{
    NormalizedSourceFile, REPEATED_BLOCK_MINIMUM_LINES, RepeatedLinePair, larger_direction_pairs,
    measure_repeated_blocks, substantial_block_starts,
};
use crate::check_duplication::report::{measurement_lines, verdict_line};
use crate::check_duplication::source_classification::{
    area_of, is_production_source, is_skipped_directory_name,
};
use crate::check_duplication::source_normalization::{
    is_substantial_code_line, mask_family_names, normalize_code_lines,
};

/// Six distinct substantial lines: one repeated block when two files hold
/// them in this order.
const SHARED_BLOCK_TEXT: &str = "let first_reading = runtime.read_first_tensor()?;
let second_reading = runtime.read_second_tensor()?;
let third_reading = runtime.read_third_tensor()?;
let fourth_reading = runtime.read_fourth_tensor()?;
let fifth_reading = runtime.read_fifth_tensor()?;
let sixth_reading = runtime.read_sixth_tensor()?;
";

fn source_file(relative_path: &str, source_text: &str) -> NormalizedSourceFile {
    NormalizedSourceFile::from_source_text(
        relative_path.to_owned(),
        area_of(relative_path),
        source_text,
    )
}

fn arguments(argument_texts: &[&str]) -> Vec<String> {
    let mut owned_arguments = Vec::new();
    for argument_text in argument_texts {
        owned_arguments.push((*argument_text).to_owned());
    }
    owned_arguments
}

fn counts(production_repeated_lines: usize, test_repeated_lines: usize) -> RepeatedLineCounts {
    RepeatedLineCounts {
        production_repeated_lines,
        test_repeated_lines,
    }
}

#[test]
fn should_drop_blank_and_comment_lines_and_trim_whitespace() {
    let code_lines = normalize_code_lines(
        "//! module doc\n\n    let counted = 1;   \n  // a comment\n/// item doc\n\tfn kept() {}\n",
    );
    assert_eq!(code_lines, ["let counted = 1;", "fn kept() {}"]);
}

#[test]
fn should_return_no_lines_for_an_empty_or_comment_only_source() {
    assert!(normalize_code_lines("").is_empty());
    assert!(normalize_code_lines("// only\n   \n//! comments\n").is_empty());
}

#[test]
fn should_mask_family_names_in_camel_and_snake_case() {
    assert_eq!(
        mask_family_names("let model: Qwen3_5ResidentModel = qwen3_5_streaming::load();"),
        "let model: FAMILYModel = FAMILY::load();"
    );
    assert_eq!(
        mask_family_names("KokoroEngine phonon_voice ClefScorer kev_head laya_trunk"),
        "FAMILYEngine FAMILY_voice FAMILYScorer FAMILY_head FAMILY_trunk"
    );
    assert_eq!(
        mask_family_names("Flux2KleinError qwen_image_21::run PrismHadamardQwen35Engine"),
        "FAMILYError FAMILY::run FAMILYEngine"
    );
}

#[test]
fn should_mask_a_compound_family_name_once_rather_than_by_its_parts() {
    assert_eq!(mask_family_names("Qwen3_5Streaming"), "FAMILY");
    assert_eq!(mask_family_names("prism_hadamard_qwen35"), "FAMILY");
}

#[test]
fn should_leave_a_line_without_family_names_unchanged_including_non_ascii_text() {
    let unchanged_line = "let résumé = \"Qwen 3\"; // naïve";
    assert_eq!(mask_family_names(unchanged_line), unchanged_line);
    assert_eq!(mask_family_names(""), "");
}

#[test]
fn should_count_only_long_lines_with_more_than_brackets_as_substantial() {
    assert!(is_substantial_code_line("let reading = runtime.read()?;"));
    assert!(!is_substantial_code_line("Ok(())"));
    assert!(!is_substantial_code_line("}); }); }); })"));
    assert!(!is_substantial_code_line(""));
}

#[test]
fn should_treat_a_twelve_character_line_as_short_and_thirteen_as_substantial() {
    assert!(!is_substantial_code_line("abcdefghijkl"));
    assert!(is_substantial_code_line("abcdefghijklm"));
}

#[test]
fn should_classify_production_and_test_sources() {
    assert!(is_production_source(
        "crates/foundation/config/src/model_family.rs"
    ));
    assert!(!is_production_source(
        "crates/foundation/config/src/model_family_tests.rs"
    ));
    assert!(!is_production_source(
        "crates/foundation/config/src/frozen_fixture_support.rs"
    ));
    assert!(!is_production_source(
        "crates/foundation/config/tests/integration/model.rs"
    ));
    assert!(is_production_source(
        "xtask/src/tests_inventory/selection.rs"
    ));
}

#[test]
fn should_skip_build_output_and_pinned_third_party_directories_only() {
    assert!(is_skipped_directory_name("target"));
    assert!(is_skipped_directory_name("third-party"));
    assert!(!is_skipped_directory_name("disposable_target"));
    assert!(!is_skipped_directory_name("src"));
}

#[test]
fn should_name_the_area_by_crate_app_and_family_folder() {
    assert_eq!(
        area_of("crates/foundation/config/src/model_discovery/kev.rs"),
        "crates/foundation/config"
    );
    assert_eq!(
        area_of("apps/daemon/src/worker/decisions_request.rs"),
        "apps/daemon"
    );
    assert_eq!(area_of("xtask/src/check_duplication.rs"), "xtask");
    assert_eq!(
        area_of("crates/families/text/qwen35/src/qwen3_5_streaming/inference_execution/mod.rs"),
        "crates/families/text/qwen35/qwen3_5_streaming"
    );
    assert_eq!(
        area_of("crates/families/text/qwen35/src/qwen3_5_core.rs"),
        "crates/families/text/qwen35/qwen3_5_core"
    );
    assert_eq!(
        area_of("crates/families/speech/speech/src/speech_models/kokoro/engine.rs"),
        "crates/families/speech/speech/kokoro"
    );
}

#[test]
fn should_keep_a_file_above_the_family_folders_in_its_crate_area() {
    assert_eq!(
        area_of("crates/families/speech/speech/src/speech_engine.rs"),
        "crates/families/speech/speech"
    );
    assert_eq!(area_of("Cargo.toml"), "Cargo.toml");
}

#[test]
fn should_find_no_block_starts_in_a_file_shorter_than_a_block() {
    assert!(substantial_block_starts(&[true; REPEATED_BLOCK_MINIMUM_LINES - 1]).is_empty());
    assert!(substantial_block_starts(&[]).is_empty());
}

#[test]
fn should_start_blocks_only_where_enough_lines_are_substantial() {
    assert_eq!(substantial_block_starts(&[true; 7]), [0, 1]);
    // Four substantial lines of six is the least a block may hold.
    assert_eq!(
        substantial_block_starts(&[true, true, false, true, true, false, false]),
        [0]
    );
    assert!(substantial_block_starts(&[true, false, false, true, true, false]).is_empty());
}

#[test]
fn should_count_a_block_shared_by_two_files_on_both_sides() {
    let measurement = measure_repeated_blocks(&[
        source_file("crates/tooling/alpha/src/one.rs", SHARED_BLOCK_TEXT),
        source_file("crates/tooling/beta/src/two.rs", SHARED_BLOCK_TEXT),
    ]);
    assert_eq!(measurement.file_count, 2);
    assert_eq!(measurement.code_line_count, 12);
    assert_eq!(measurement.repeated_line_count, 12);
    assert_eq!(
        measurement.area_pairs,
        [RepeatedLinePair {
            first_place: "crates/tooling/alpha".to_owned(),
            second_place: "crates/tooling/beta".to_owned(),
            repeated_line_count: 6,
        }]
    );
    assert_eq!(measurement.file_pairs[0].repeated_line_count, 6);
}

#[test]
fn should_count_copies_that_differ_only_by_family_name() {
    let resident_copy = SHARED_BLOCK_TEXT.replace("runtime", "resident_runtime");
    let streaming_copy = SHARED_BLOCK_TEXT.replace("runtime", "streaming_runtime");
    let measurement = measure_repeated_blocks(&[
        source_file(
            "crates/families/text/qwen35/src/qwen3_5_resident/load.rs",
            &resident_copy,
        ),
        source_file(
            "crates/families/text/qwen35/src/qwen3_5_streaming/load.rs",
            &streaming_copy,
        ),
    ]);
    assert_eq!(measurement.repeated_line_count, 12);
}

#[test]
fn should_not_count_a_block_repeated_inside_one_file_or_shorter_than_six_lines() {
    let twice_in_one_file = format!("{SHARED_BLOCK_TEXT}fn separator() {{}}\n{SHARED_BLOCK_TEXT}");
    let five_lines: String = SHARED_BLOCK_TEXT
        .lines()
        .take(5)
        .collect::<Vec<_>>()
        .join("\n");
    let measurement = measure_repeated_blocks(&[
        source_file("crates/tooling/alpha/src/one.rs", &twice_in_one_file),
        source_file("crates/tooling/beta/src/two.rs", &five_lines),
    ]);
    assert_eq!(measurement.repeated_line_count, 0);
    assert!(measurement.area_pairs.is_empty());
    assert!(measurement.file_pairs.is_empty());
}

#[test]
fn should_not_count_shared_runs_of_closing_brackets() {
    let closing_brackets = "}\n});\n}\n)\n]\n};\n}\n";
    let measurement = measure_repeated_blocks(&[
        source_file("crates/tooling/alpha/src/one.rs", closing_brackets),
        source_file("crates/tooling/beta/src/two.rs", closing_brackets),
    ]);
    assert_eq!(measurement.repeated_line_count, 0);
}

#[test]
fn should_report_an_empty_corpus_as_zero_percent() {
    let measurement = measure_repeated_blocks(&[]);
    assert_eq!(measurement.code_line_count, 0);
    assert!(measurement.repeated_percentage().abs() < f64::EPSILON);
}

#[test]
fn should_count_a_line_shared_with_two_files_once_in_the_total() {
    let measurement = measure_repeated_blocks(&[
        source_file("crates/tooling/alpha/src/one.rs", SHARED_BLOCK_TEXT),
        source_file("crates/tooling/beta/src/two.rs", SHARED_BLOCK_TEXT),
        source_file("crates/tooling/beta/src/three.rs", SHARED_BLOCK_TEXT),
    ]);
    assert_eq!(measurement.repeated_line_count, 18);
    let alpha_beta_pair = &measurement.area_pairs[0];
    assert_eq!(alpha_beta_pair.first_place, "crates/tooling/beta");
    assert_eq!(alpha_beta_pair.second_place, "crates/tooling/alpha");
    assert_eq!(alpha_beta_pair.repeated_line_count, 12);
}

#[test]
fn should_keep_the_larger_direction_and_sort_pairs_largest_first() {
    let place_names = ["first.rs", "second.rs", "third.rs"];
    let mut directional_counts = HashMap::new();
    directional_counts.insert((0, 1), 4);
    directional_counts.insert((1, 0), 9);
    directional_counts.insert((2, 2), 20);
    let repeated_line_pairs = larger_direction_pairs(&directional_counts, &place_names);
    assert_eq!(repeated_line_pairs.len(), 2);
    assert_eq!(repeated_line_pairs[0].repeated_line_count, 20);
    assert_eq!(repeated_line_pairs[1].first_place, "second.rs");
    assert_eq!(repeated_line_pairs[1].repeated_line_count, 9);
}

#[test]
fn should_parse_and_render_a_baseline_round_trip() {
    let baseline_text = render_baseline(counts(14_912, 10_052));
    assert!(baseline_text.ends_with("}\n"));
    assert_eq!(parse_baseline(&baseline_text), Ok(counts(14_912, 10_052)));
}

#[test]
fn should_refuse_a_baseline_with_a_missing_or_negative_count() {
    assert!(parse_baseline("{\"production_repeated_lines\": 5}").is_err());
    assert!(
        parse_baseline("{\"production_repeated_lines\": -1, \"test_repeated_lines\": 0}").is_err()
    );
    assert!(parse_baseline("not json").is_err());
}

#[test]
fn should_pass_a_held_or_lowered_count_and_fail_a_grown_one() {
    let corpus_verdicts = judge_against_baseline(counts(100, 50), counts(100, 49));
    assert_eq!(corpus_verdicts[0].comparison, CorpusComparison::Held);
    assert_eq!(corpus_verdicts[1].comparison, CorpusComparison::Lowered);
    assert!(!has_grown_corpus(&corpus_verdicts));
    let grown_verdicts = judge_against_baseline(counts(100, 50), counts(101, 0));
    assert_eq!(grown_verdicts[0].comparison, CorpusComparison::Grown);
    assert!(has_grown_corpus(&grown_verdicts));
}

#[test]
fn should_skip_judgement_only_when_the_ref_has_no_baseline() {
    assert_eq!(judge_committed_baseline(None, counts(1, 1)), Ok(None));
    assert!(judge_committed_baseline(Some("{}"), counts(1, 1)).is_err());
    let committed_text = render_baseline(counts(2, 2));
    let corpus_verdicts = judge_committed_baseline(Some(&committed_text), counts(3, 2));
    assert!(has_grown_corpus(&corpus_verdicts.unwrap().unwrap()));
}

#[test]
fn should_tell_an_absent_baseline_from_an_unknown_ref() {
    assert!(is_absent_at_ref_message(
        "fatal: path 'duplication-baseline.json' does not exist in 'HEAD'"
    ));
    assert!(is_absent_at_ref_message(
        "fatal: path 'duplication-baseline.json' exists on disk, but not in 'origin/main'"
    ));
    assert!(!is_absent_at_ref_message(
        "fatal: invalid object name 'no-such-ref'."
    ));
}

#[test]
fn should_parse_the_default_and_every_flag() {
    assert_eq!(
        DuplicationInvocation::parse(&[]),
        Ok(DuplicationInvocation::default())
    );
    let invocation = DuplicationInvocation::parse(&arguments(&[
        "--write",
        "--against",
        "origin/main",
        "--top",
        "3",
    ]))
    .unwrap();
    assert!(invocation.should_write_baseline);
    assert_eq!(invocation.baseline_ref, "origin/main");
    assert_eq!(invocation.top_pair_count, 3);
}

#[test]
fn should_refuse_unknown_arguments_and_missing_or_malformed_values() {
    assert!(DuplicationInvocation::parse(&arguments(&["--fast"])).is_err());
    assert!(DuplicationInvocation::parse(&arguments(&["--against"])).is_err());
    assert!(DuplicationInvocation::parse(&arguments(&["--against", "--write"])).is_err());
    assert!(DuplicationInvocation::parse(&arguments(&["--top", "many"])).is_err());
}

#[test]
fn should_print_totals_and_at_most_the_requested_pairs() {
    let measurement = measure_repeated_blocks(&[
        source_file("crates/tooling/alpha/src/one.rs", SHARED_BLOCK_TEXT),
        source_file("crates/tooling/beta/src/two.rs", SHARED_BLOCK_TEXT),
        source_file("crates/tooling/beta/src/three.rs", SHARED_BLOCK_TEXT),
    ]);
    let report_lines = measurement_lines("production", &measurement, 1);
    assert_eq!(
        report_lines[0],
        "[check-duplication] corpus=production files=3 code_lines=18 repeated_lines=18 repeated_percentage=100.0"
    );
    assert_eq!(report_lines.len(), 5);
    assert!(report_lines[2].contains("crates/tooling/beta  <->  crates/tooling/alpha"));
    assert_eq!(measurement_lines("production", &measurement, 0).len(), 1);
}

#[test]
fn should_name_the_remedy_for_a_grown_or_lowered_corpus_only() {
    let corpus_verdicts = judge_against_baseline(counts(10, 10), counts(11, 9));
    let grown_line = verdict_line(&corpus_verdicts[0], "HEAD");
    assert!(grown_line.starts_with("[check-duplication] verdict=grown corpus=production"));
    assert!(grown_line.contains("hint="));
    assert!(verdict_line(&corpus_verdicts[1], "HEAD").contains("--write"));
    let held_verdicts = judge_against_baseline(counts(10, 10), counts(10, 10));
    assert!(!verdict_line(&held_verdicts[0], "HEAD").contains("hint="));
}

#[test]
fn should_report_progress_with_percentage_and_eta() {
    assert_eq!(
        progress_line(250, 1000, Duration::from_secs(5)),
        "[check-duplication] status=progress step=read-sources files=250/1000 percentage=25.0 elapsed_seconds=5.0 ETA_seconds=15.0"
    );
}

#[test]
fn should_report_zero_eta_before_any_file_and_full_percentage_for_no_files() {
    assert!(progress_line(0, 10, Duration::from_secs(1)).ends_with("ETA_seconds=0.0"));
    assert!(progress_line(0, 0, Duration::ZERO).contains("percentage=100.0"));
}

#[test]
fn should_report_progress_only_once_the_interval_has_passed() {
    assert!(!is_progress_report_due(Duration::from_millis(4_999)));
    assert!(is_progress_report_due(Duration::from_secs(5)));
}
