use astronomical_model_serving::Flux2KleinFlowScheduler;

#[test]
fn should_build_four_cpu_fp64_flow_euler_steps_deterministically() {
    let schedule = Flux2KleinFlowScheduler::schedule(1_024, 1_024)
        .expect("official dimensions should produce a schedule");

    assert_eq!(schedule.steps().len(), 4);
    assert_eq!(schedule.initial_sigma(), schedule.steps()[0].sigma());
    assert_eq!(schedule.steps()[3].next_sigma(), 0.0);
    for adjacent_steps in schedule.steps().windows(2) {
        assert_eq!(adjacent_steps[0].next_sigma(), adjacent_steps[1].sigma());
        assert!(adjacent_steps[0].delta_sigma() < 0.0);
    }
    assert!(schedule.steps()[3].delta_sigma() < 0.0);

    let repeated = Flux2KleinFlowScheduler::schedule(1_024, 1_024)
        .expect("the repeated schedule should build");
    assert_eq!(schedule, repeated);
}

#[test]
fn should_shift_the_schedule_from_the_packed_image_sequence_length() {
    let small = Flux2KleinFlowScheduler::schedule(512, 512)
        .expect("small aligned dimensions should schedule");
    let large = Flux2KleinFlowScheduler::schedule(1_024, 1_024)
        .expect("large aligned dimensions should schedule");

    assert_eq!(small.image_sequence_length(), 1_024);
    assert_eq!(large.image_sequence_length(), 4_096);
    assert_eq!(large.initial_sigma(), 1.0);
    assert_eq!(small.initial_sigma(), 1.0);
    assert!(large.steps()[1].sigma() > small.steps()[1].sigma());
}
