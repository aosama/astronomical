use super::{QueueDiskShortfall, admit_into_queue};

const GIBIBYTE: u64 = 1 << 30;

#[test]
fn should_admit_a_model_when_the_whole_list_fits_with_its_margin() {
    let admission = admit_into_queue(
        100 * GIBIBYTE,
        10 * GIBIBYTE,
        &[20 * GIBIBYTE],
        30 * GIBIBYTE,
    );

    assert_eq!(admission, Ok(()));
}

#[test]
fn should_refuse_a_model_when_the_list_and_margin_exceed_the_free_space() {
    let admission = admit_into_queue(
        60 * GIBIBYTE,
        10 * GIBIBYTE,
        &[20 * GIBIBYTE],
        30 * GIBIBYTE,
    );

    // 60 GiB listed plus its one-percent margin is just over 60 GiB.
    assert_eq!(
        admission,
        Err(QueueDiskShortfall {
            required_bytes: 60 * GIBIBYTE + (60 * GIBIBYTE).div_ceil(100),
            available_bytes: 60 * GIBIBYTE,
        })
    );
}

#[test]
fn should_keep_at_least_one_byte_of_margin_for_a_tiny_list() {
    assert!(admit_into_queue(10, 0, &[], 10).is_err());
    assert_eq!(admit_into_queue(11, 0, &[], 10), Ok(()));
}

#[test]
fn should_refuse_rather_than_overflow_on_impossible_sizes() {
    let admission = admit_into_queue(u64::MAX - 1, u64::MAX, &[u64::MAX], u64::MAX);

    assert_eq!(
        admission,
        Err(QueueDiskShortfall {
            required_bytes: u64::MAX,
            available_bytes: u64::MAX - 1,
        })
    );
}
