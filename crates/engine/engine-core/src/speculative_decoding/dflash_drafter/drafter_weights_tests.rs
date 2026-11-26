use super::drafter_load_peak_bytes;

#[test]
fn should_ask_for_three_eighths_of_a_drafter_checkpoint() {
    let dense_drafter_checkpoint_bytes = 3_843_000_000;

    assert_eq!(
        drafter_load_peak_bytes(dense_drafter_checkpoint_bytes),
        1_441_125_000
    );
}

#[test]
fn should_round_a_partial_eighth_up_rather_than_ask_for_too_little() {
    assert_eq!(drafter_load_peak_bytes(9), 6, "two eighths rounded up, times three");
    assert_eq!(drafter_load_peak_bytes(1), 3);
}

#[test]
fn should_ask_for_nothing_for_an_empty_checkpoint() {
    assert_eq!(drafter_load_peak_bytes(0), 0);
}
