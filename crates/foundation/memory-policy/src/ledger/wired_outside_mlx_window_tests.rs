use super::{RELEASE_LAG_WINDOW_MILLISECONDS, WiredOutsideMlxWindow};

const GIBIBYTE: u64 = 1 << 30;

#[test]
fn should_take_the_first_reading_as_it_is() {
    let mut window = WiredOutsideMlxWindow::default();
    assert_eq!(window.observe(0, 4 * GIBIBYTE), 4 * GIBIBYTE);
    assert_eq!(window.latest_wired_outside_mlx_bytes(), Some(4 * GIBIBYTE));
}

#[test]
fn should_not_count_a_rise_that_ends_within_the_release_lag() {
    // Readings every 50 ms; a release inflates them for 250 ms.
    let mut window = WiredOutsideMlxWindow::default();
    assert_eq!(window.observe(0, 4 * GIBIBYTE), 4 * GIBIBYTE);
    for reading_milliseconds in [50, 100, 150, 200, 250] {
        assert_eq!(
            window.observe(reading_milliseconds, 6 * GIBIBYTE),
            4 * GIBIBYTE,
            "an inflated reading at {reading_milliseconds} ms must not count"
        );
    }
    assert_eq!(window.observe(300, 4 * GIBIBYTE), 4 * GIBIBYTE);
}

#[test]
fn should_count_a_rise_once_it_has_lasted_the_whole_window() {
    let mut window = WiredOutsideMlxWindow::default();
    let _ = window.observe(0, 4 * GIBIBYTE);
    let _ = window.observe(50, 6 * GIBIBYTE);
    assert_eq!(
        window.observe(50 + RELEASE_LAG_WINDOW_MILLISECONDS - 1, 6 * GIBIBYTE),
        4 * GIBIBYTE
    );
    assert_eq!(
        window.observe(50 + RELEASE_LAG_WINDOW_MILLISECONDS, 6 * GIBIBYTE),
        6 * GIBIBYTE,
        "at exactly the window's length the old reading has left it"
    );
}

#[test]
fn should_count_a_fall_at_once() {
    let mut window = WiredOutsideMlxWindow::default();
    let _ = window.observe(0, 6 * GIBIBYTE);
    assert_eq!(window.observe(10, 3 * GIBIBYTE), 3 * GIBIBYTE);
}

#[test]
fn should_keep_the_last_reading_before_a_long_gap() {
    // Idle for seconds, then a reading inflated by a release at a prompt's
    // start: the reading before the gap still stands for the true figure.
    let mut window = WiredOutsideMlxWindow::default();
    let _ = window.observe(0, 4 * GIBIBYTE);
    assert_eq!(window.observe(5_000, 7 * GIBIBYTE), 4 * GIBIBYTE);
    assert_eq!(window.latest_reading_milliseconds(), Some(5_000));
}

#[test]
fn should_report_no_latest_reading_before_any() {
    let window = WiredOutsideMlxWindow::default();
    assert_eq!(window.latest_wired_outside_mlx_bytes(), None);
    assert_eq!(window.latest_reading_milliseconds(), None);
}

#[test]
fn should_take_every_reading_as_read_with_no_window() {
    let mut window = WiredOutsideMlxWindow::with_window_milliseconds(0);
    let _ = window.observe(0, 4 * GIBIBYTE);
    assert_eq!(window.observe(50, 6 * GIBIBYTE), 6 * GIBIBYTE);
}
