use astronomical_ipc_protocol::ModelLoadInFlight;

use super::ModelLoadWatch;

fn load_report(model_id: &str) -> ModelLoadInFlight {
    ModelLoadInFlight::for_model(model_id)
}

#[test]
fn should_treat_the_first_report_as_a_new_load() {
    let mut model_load_watch = ModelLoadWatch::default();

    assert!(model_load_watch.reports_a_new_load(&load_report("first-model")));
}

#[test]
fn should_treat_another_model_as_a_new_load() {
    let mut model_load_watch = ModelLoadWatch::default();
    model_load_watch.reports_a_new_load(&load_report("first-model"));

    assert!(model_load_watch.reports_a_new_load(&load_report("second-model")));
}

#[test]
fn should_not_treat_a_repeated_report_as_a_new_load() {
    let mut model_load_watch = ModelLoadWatch::default();
    model_load_watch.reports_a_new_load(&load_report("first-model"));

    assert!(!model_load_watch.reports_a_new_load(&load_report("first-model")));
}

#[test]
fn should_pad_a_shorter_line_over_a_longer_one() {
    let mut model_load_watch = ModelLoadWatch::default();
    let mut stderr = Vec::new();

    model_load_watch.draw_live_line("twelve chars", &mut stderr);
    model_load_watch.draw_live_line("short", &mut stderr);

    assert_eq!(
        String::from_utf8(stderr).expect("the line is text"),
        "\rtwelve chars\rshort       "
    );
}

#[test]
fn should_end_a_live_line_exactly_once() {
    let mut model_load_watch = ModelLoadWatch::default();
    let mut stderr = Vec::new();

    model_load_watch.draw_live_line("line", &mut stderr);
    model_load_watch.finish_live_line(&mut stderr);
    model_load_watch.finish_live_line(&mut stderr);

    assert_eq!(
        String::from_utf8(stderr).expect("the line is text"),
        "\rline\n"
    );
}

#[test]
fn should_write_nothing_when_no_line_was_ever_drawn() {
    let mut model_load_watch = ModelLoadWatch::default();
    let mut stderr = Vec::new();

    model_load_watch.finish_live_line(&mut stderr);

    assert!(
        stderr.is_empty(),
        "a fast answer must not gain a blank line"
    );
}
