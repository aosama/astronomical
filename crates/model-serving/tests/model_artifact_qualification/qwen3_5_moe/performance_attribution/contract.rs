use std::time::Duration;

use tokio::time::interval;

use crate::common::generation_progress::await_generation_advance_with_live_progress;

#[tokio::test]
async fn should_keep_one_generation_advance_in_flight_while_reporting_progress() {
    let mut progress_interval = interval(Duration::from_millis(5));
    progress_interval.tick().await;
    let mut reported_progress_count = 0_u32;
    let generation_advance_outcome = await_generation_advance_with_live_progress(
        async {
            tokio::time::sleep(Duration::from_millis(20)).await;
            42_u32
        },
        &mut progress_interval,
        || reported_progress_count += 1,
    )
    .await;
    assert_eq!(generation_advance_outcome, 42);
    assert!(reported_progress_count > 0);
}
