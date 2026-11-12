use super::unknown_voice_reason;

// Positive: the reason names the voice asked for and every voice offered.
#[test]
fn should_name_the_missing_voice_and_the_voices_offered() {
    let reason = unknown_voice_reason("alloy", &["af_heart".to_owned(), "am_michael".to_owned()]);
    assert_eq!(
        reason,
        "there is no voice \"alloy\"; the voices are af_heart, am_michael"
    );
}

// Edge: a model with no voices still yields a readable reason.
#[test]
fn should_read_well_for_a_model_without_voices() {
    assert!(unknown_voice_reason("alloy", &[]).ends_with("the voices are "));
}
