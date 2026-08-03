use astronomical_inference_worker::worker_startup::astronomical_log_rotation;

#[test]
fn should_rotate_worker_logs_every_hour() {
    assert_eq!(
        astronomical_log_rotation(),
        tracing_appender::rolling::Rotation::HOURLY
    );
}
