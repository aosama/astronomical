use super::*;

use config::ProblemReportChoice;

fn config_with_sharing(sharing_section: &str) -> String {
    format!(
        r#"{{"$schema":"./astronomical-config.schema.json","schema_version":1,"runtime":{{"model_directories":[]}},"sharing":{sharing_section}}}"#
    )
}

fn load_config(document: &str) -> Result<AstronomicalConfig, config::AstronomicalConfigError> {
    let temporary_home_directory = tempfile::tempdir().expect("temporary home should be created");
    write_config(temporary_home_directory.path(), document);
    AstronomicalConfig::load_from_home_directory(temporary_home_directory.path())
}

/// Nothing is sent until the person chooses: reports are asked about each
/// time, and the daily count question stays unanswered.
#[test]
fn should_ask_about_reports_and_leave_the_daily_count_unanswered_when_omitted() {
    let temporary_home_directory = tempfile::tempdir().expect("temporary home should be created");
    let astronomical_config =
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path())
            .expect("missing config should load");
    assert_eq!(
        astronomical_config.problem_report_choice(),
        ProblemReportChoice::Ask
    );
    assert_eq!(astronomical_config.daily_count_answer(), None);
}

#[test]
fn should_read_each_report_choice_and_both_daily_count_answers() {
    for (sharing_section, expected_choice, expected_answer) in [
        (
            r#"{"problem_reports":"always"}"#,
            ProblemReportChoice::Always,
            None,
        ),
        (
            r#"{"problem_reports":"never"}"#,
            ProblemReportChoice::Never,
            None,
        ),
        (
            r#"{"problem_reports":"ask","daily_count":true}"#,
            ProblemReportChoice::Ask,
            Some(true),
        ),
        (
            r#"{"daily_count":false}"#,
            ProblemReportChoice::Ask,
            Some(false),
        ),
        ("{}", ProblemReportChoice::Ask, None),
    ] {
        let astronomical_config =
            load_config(&config_with_sharing(sharing_section)).expect("config should load");
        assert_eq!(
            (
                astronomical_config.problem_report_choice(),
                astronomical_config.daily_count_answer()
            ),
            (expected_choice, expected_answer),
            "{sharing_section}"
        );
    }
}

#[test]
fn should_refuse_an_unknown_report_choice_or_setting() {
    for sharing_section in [
        r#"{"problem_reports":"sometimes"}"#,
        r#"{"problem_reports":true}"#,
        r#"{"daily_count":"yes"}"#,
        r#"{"crash_reports":true}"#,
    ] {
        assert!(
            load_config(&config_with_sharing(sharing_section)).is_err(),
            "{sharing_section} should be refused"
        );
    }
}

/// Answering the daily count question must never replace the worker and
/// unload the person's model.
#[test]
fn should_leave_the_configuration_generation_unchanged_by_the_sharing_section() {
    let without_section = load_config(&config_with_sharing("{}")).expect("config should load");
    let with_section = load_config(&config_with_sharing(
        r#"{"problem_reports":"always","daily_count":true}"#,
    ))
    .expect("config should load");
    assert_eq!(without_section.generation(), with_section.generation());
}

#[test]
fn should_match_the_schema_defaults_and_list_both_settings_on_the_settings_page() {
    let schema_document: serde_json::Value =
        serde_json::from_slice(config::config_schema_bytes()).expect("schema should be JSON");
    assert_eq!(
        schema_document["properties"]["sharing"]["$ref"],
        "#/$defs/sharing"
    );
    let sharing_properties = &schema_document["$defs"]["sharing"]["properties"];
    assert_eq!(sharing_properties["problem_reports"]["default"], "ask");
    assert_eq!(sharing_properties["daily_count"]["default"], false);
    let layout_groups = schema_document["x-astronomical-layout"]["settings"]
        .as_array()
        .expect("layout should list setting groups");
    let mut is_listed = false;
    for layout_group in layout_groups {
        if layout_group["title"] == "What leaves this Mac" {
            is_listed = layout_group["settings"]
                == serde_json::json!(["sharing.problem_reports", "sharing.daily_count"]);
        }
    }
    assert!(
        is_listed,
        "the What leaves this Mac group should list both settings"
    );
}
