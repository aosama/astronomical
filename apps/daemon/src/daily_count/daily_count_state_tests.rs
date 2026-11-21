use super::{DAILY_COUNT_STATE_FILE_NAME, DailyCountState};

#[test]
fn should_start_today_and_read_the_same_state_back() {
    let state_directory = tempfile::tempdir().expect("temporary folder");
    let started_state = DailyCountState::load_or_start(state_directory.path(), 20_000);
    assert_eq!(started_state.first_used_unix_day, 20_000);
    assert_eq!(started_state.last_sent_unix_day, None);
    let read_state = DailyCountState::load_or_start(state_directory.path(), 20_005);
    assert_eq!(
        read_state, started_state,
        "a later day must not move the first day"
    );
}

#[test]
fn should_start_again_when_the_state_file_is_damaged() {
    let state_directory = tempfile::tempdir().expect("temporary folder");
    std::fs::write(
        state_directory.path().join(DAILY_COUNT_STATE_FILE_NAME),
        b"{not json",
    )
    .expect("damaged state should be written");
    let restarted_state = DailyCountState::load_or_start(state_directory.path(), 20_010);
    assert_eq!(restarted_state.first_used_unix_day, 20_010);
}

#[test]
fn should_ask_only_while_unanswered_and_only_after_the_first_day() {
    let daily_count_state = DailyCountState {
        first_used_unix_day: 20_000,
        last_sent_unix_day: None,
    };
    assert!(
        !daily_count_state.should_ask(None, 20_000),
        "never on the first day"
    );
    assert!(daily_count_state.should_ask(None, 20_001));
    assert!(!daily_count_state.should_ask(Some(true), 20_001));
    assert!(
        !daily_count_state.should_ask(Some(false), 20_001),
        "No thanks is final"
    );
    assert!(
        !daily_count_state.should_ask(None, 19_999),
        "a clock set back must not ask early"
    );
}

#[test]
fn should_know_whether_a_count_went_today() {
    let daily_count_state = DailyCountState {
        first_used_unix_day: 20_000,
        last_sent_unix_day: Some(20_003),
    };
    assert!(daily_count_state.is_sent_today(20_003));
    assert!(!daily_count_state.is_sent_today(20_004));
}
