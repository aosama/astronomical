use super::request_generation_settings_presence;

#[test]
fn should_record_only_the_settings_the_request_supplied() {
    let settings_presence = request_generation_settings_presence(Some(256), None, Some(0.9));
    assert!(settings_presence.maximum_output_tokens);
    assert!(!settings_presence.temperature);
    assert!(settings_presence.top_p);
    assert!(!settings_presence.repetition_penalty);
}

#[test]
fn should_record_no_settings_for_a_request_that_supplied_none() {
    let settings_presence = request_generation_settings_presence(None, None, None);
    assert!(!settings_presence.maximum_output_tokens);
    assert!(!settings_presence.temperature);
    assert!(!settings_presence.top_p);
    assert!(!settings_presence.repetition_penalty);
}

#[test]
fn should_never_record_repetition_penalty_as_request_supplied() {
    let settings_presence = request_generation_settings_presence(Some(1), Some(0.0), Some(1.0));
    assert!(settings_presence.temperature);
    assert!(!settings_presence.repetition_penalty);
}
