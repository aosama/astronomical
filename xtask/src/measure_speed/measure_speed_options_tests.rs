use super::{MeasureSpeedOptions, describe_mac};

fn arguments(argument_texts: &[&str]) -> Vec<String> {
    argument_texts
        .iter()
        .map(|argument_text| (*argument_text).to_owned())
        .collect()
}

#[test]
fn should_read_every_option_and_keep_models_in_order() {
    let parsed_options = MeasureSpeedOptions::parse(&arguments(&[
        "--rounds",
        "5",
        "--measured-on",
        "Mac with an M4 Max and 128 GB",
        "--write-catalog",
        "--temperature",
        "0.7",
        "--base-url",
        "http://127.0.0.1:9999",
        "Model-B",
        "Model-A",
    ]))
    .expect("valid options");

    assert_eq!(
        parsed_options,
        MeasureSpeedOptions {
            base_url: "http://127.0.0.1:9999".to_owned(),
            round_count: 5,
            temperature: 0.7,
            measured_on: Some("Mac with an M4 Max and 128 GB".to_owned()),
            should_write_catalog: true,
            model_ids: vec!["Model-B".to_owned(), "Model-A".to_owned()],
        }
    );
}

#[test]
fn should_default_to_development_three_rounds_and_no_catalog_write() {
    let parsed_options = MeasureSpeedOptions::parse(&arguments(&["Model-A"])).expect("valid");

    assert_eq!(
        parsed_options.base_url, "http://127.0.0.1:6733",
        "Development, never Stable"
    );
    assert_eq!(parsed_options.round_count, 3);
    assert_eq!(
        parsed_options.temperature, 0.6,
        "the temperature people chat at"
    );
    assert!(!parsed_options.should_write_catalog);
    assert_eq!(
        parsed_options.measured_on, None,
        "parsing leaves the Mac's description to the run"
    );
}

#[test]
fn should_refuse_no_models_too_few_rounds_unknown_options_and_missing_values() {
    for refused_arguments in [
        vec![],
        vec!["--rounds", "2", "Model-A"],
        vec!["--rounds", "three", "Model-A"],
        vec!["--fast", "Model-A"],
        vec!["Model-A", "--base-url"],
        vec!["--temperature", "0", "Model-A"],
        vec!["--temperature", "2.5", "Model-A"],
        vec!["--temperature", "warm", "Model-A"],
    ] {
        assert!(
            MeasureSpeedOptions::parse(&arguments(&refused_arguments)).is_err(),
            "{refused_arguments:?}"
        );
    }
}

#[test]
fn should_describe_a_mac_with_the_right_article() {
    assert_eq!(describe_mac("M5 Pro", 48), "Mac with an M5 Pro and 48 GB");
    assert_eq!(describe_mac("Mac", 8), "Mac with a Mac and 8 GB");
    assert_eq!(describe_mac("A18 Pro", 8), "Mac with an A18 Pro and 8 GB");
    assert_eq!(describe_mac("", 16), "Mac with a  and 16 GB");
}

#[test]
fn should_accept_the_temperature_bounds_of_a_chat_request() {
    for (temperature_text, expected_temperature) in [("2", 2.0), ("0.01", 0.01), ("1.0", 1.0)] {
        let parsed_options =
            MeasureSpeedOptions::parse(&arguments(&["--temperature", temperature_text, "Model-A"]))
                .expect("a temperature within the chat bounds");
        assert_eq!(parsed_options.temperature, expected_temperature);
    }
}
