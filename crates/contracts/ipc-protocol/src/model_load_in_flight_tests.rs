use super::ModelLoadInFlight;

#[test]
fn should_name_no_model_until_told_which() {
    assert_eq!(ModelLoadInFlight::unnamed().model_id, None);
}

#[test]
fn should_name_the_model_it_was_given() {
    assert_eq!(
        ModelLoadInFlight::for_model("next-model")
            .model_id
            .as_deref(),
        Some("next-model")
    );
}

#[test]
fn should_survive_the_wire_unchanged() {
    let report = ModelLoadInFlight::for_model("next-model");
    let wire_text = serde_json::to_string(&report).expect("the report serializes");

    assert_eq!(wire_text, r#"{"model_id":"next-model"}"#);
    assert_eq!(
        serde_json::from_str::<ModelLoadInFlight>(&wire_text).expect("the report parses"),
        report
    );
}

#[test]
fn should_reject_an_unknown_field() {
    assert!(
        serde_json::from_str::<ModelLoadInFlight>(r#"{"model_id":null,"stage":"weights"}"#)
            .is_err()
    );
}
