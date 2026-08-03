use astronomical_supervisor::astronomical_log_rotation;

#[test]
fn should_rotate_supervisor_logs_every_hour() {
    assert_eq!(
        astronomical_log_rotation(),
        tracing_appender::rolling::Rotation::HOURLY
    );
}
