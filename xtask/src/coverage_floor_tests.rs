use crate::coverage_floor::{
    CoverageFloorInvocation, CoverageSummary, DEFAULT_MINIMUM_LINE_COVERAGE_PERCENT,
    DEFAULT_PACKAGE_NAME, parse_summary_total_row,
};

/// A real cargo-llvm-cov summary excerpt, captured from this repository.
const REAL_SUMMARY_TABLE: &str = "\
Filename                                 Regions    Missed Regions     Cover   Functions  Missed Functions  Executed       Lines      Missed Lines     Cover    Branches   Missed Branches     Cover
admission/rotating.rs                         16                 0   100.00%           3                 0   100.00%          23                 0   100.00%           0                 0         -
budget/ram_geometry.rs                        43                43     0.00%           6                 6     0.00%          61                61     0.00%           0                 0         -
residency/page_commit.rs                      52                 1    98.08%           7                 0   100.00%          69                 1    98.55%           0                 0         -
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
TOTAL                                       1745               256    85.33%         235                32    86.38%        2367               280    88.17%           0                 0         -
";

#[test]
fn should_parse_the_total_row_of_a_real_summary_table() {
    let coverage_summary =
        parse_summary_total_row(REAL_SUMMARY_TABLE).expect("the real summary table should parse");
    assert_eq!(
        coverage_summary,
        CoverageSummary {
            region_coverage_percent: 85.33,
            function_coverage_percent: 86.38,
            line_coverage_percent: 88.17,
        }
    );
}

#[test]
fn should_fail_closed_when_the_summary_has_no_total_row() {
    let malformed_summary = "Filename Regions Cover\nsome/file.rs 4 50.00%\n";
    assert!(parse_summary_total_row(malformed_summary).is_err());
}

#[test]
fn should_fail_closed_when_the_total_row_is_too_short() {
    let short_total_row = "TOTAL 1745 256 85.33%\n";
    let parsed_summary = parse_summary_total_row(short_total_row);
    assert!(parsed_summary.is_err());
}

#[test]
fn should_fail_closed_when_a_percentage_cell_is_malformed() {
    let malformed_total_row = "TOTAL 1745 256 85.33% 235 32 86.38% 2367 280 eighty-eight 0 0 -\n";
    assert!(parse_summary_total_row(malformed_total_row).is_err());
}

#[test]
fn should_flag_only_strictly_below_floor_line_coverage() {
    let measured_summary = CoverageSummary {
        region_coverage_percent: 85.0,
        function_coverage_percent: 85.0,
        line_coverage_percent: 80.0,
    };
    assert!(
        !measured_summary.is_below_floor(80.0),
        "coverage exactly at the floor passes"
    );
    let one_hundredth_below = CoverageSummary {
        line_coverage_percent: 79.99,
        ..measured_summary
    };
    assert!(one_hundredth_below.is_below_floor(80.0));
}

#[test]
fn should_default_to_the_memory_policy_package_at_eighty_percent() {
    let default_invocation = CoverageFloorInvocation::parse(&[])
        .expect("an empty argument list should parse to the defaults");
    assert_eq!(default_invocation.package_name, DEFAULT_PACKAGE_NAME);
    assert_eq!(
        default_invocation.minimum_line_coverage_percent,
        DEFAULT_MINIMUM_LINE_COVERAGE_PERCENT
    );
}

#[test]
fn should_parse_an_override_package_and_floor() {
    let invocation = CoverageFloorInvocation::parse(&[
        "--package".to_owned(),
        "astronomical-config".to_owned(),
        "--minimum-percent".to_owned(),
        "42.5".to_owned(),
    ])
    .expect("both overrides should parse");
    assert_eq!(invocation.package_name, "astronomical-config");
    assert_eq!(invocation.minimum_line_coverage_percent, 42.5);
}

#[test]
fn should_reject_unknown_arguments_and_missing_values() {
    assert!(
        CoverageFloorInvocation::parse(&["--unexpected".to_owned(), "value".to_owned()]).is_err()
    );
    assert!(CoverageFloorInvocation::parse(&["--minimum-percent".to_owned()]).is_err());
    assert!(CoverageFloorInvocation::parse(&["--package".to_owned()]).is_err());
}

#[test]
fn should_reject_non_numeric_and_out_of_range_floors() {
    assert!(
        CoverageFloorInvocation::parse(&["--minimum-percent".to_owned(), "eighty".to_owned()])
            .is_err()
    );
    assert!(
        CoverageFloorInvocation::parse(&["--minimum-percent".to_owned(), "100.1".to_owned()])
            .is_err()
    );
    assert!(
        CoverageFloorInvocation::parse(&["--minimum-percent".to_owned(), "-1".to_owned()]).is_err()
    );
}

#[test]
fn should_fail_closed_on_not_a_number_and_infinite_floors() {
    // NaN and infinity parse as f64 but fall outside the 0..=100 range
    // check, so the gate fail-closes instead of comparing against a
    // non-orderable floor.
    assert!(
        CoverageFloorInvocation::parse(&["--minimum-percent".to_owned(), "NaN".to_owned()])
            .is_err()
    );
    assert!(
        CoverageFloorInvocation::parse(&["--minimum-percent".to_owned(), "inf".to_owned()])
            .is_err()
    );
}

#[test]
fn should_fail_closed_on_a_missing_percentage_cell() {
    // A coverage row without measured data prints `-` for the percentage;
    // that cell must fail the parse rather than count as zero.
    let missing_cell_row = "TOTAL 10 0 100.00% 2 0 100.00% 5 0 - 0 0 -\n";
    assert!(parse_summary_total_row(missing_cell_row).is_err());
}
