use super::release_speed_section;
use crate::release_speed::speed_ledger::DrafterInUse;
use crate::release_speed::speed_ledger::speed_ledger_tests::{ledger, measurement, spread};

fn row_for<'section>(section_text: &'section str, model_id: &str) -> &'section str {
    section_text
        .lines()
        .find(|line| line.starts_with(&format!("| {model_id},")))
        .unwrap_or_else(|| panic!("a row for {model_id} in:\n{section_text}"))
}

#[test]
fn should_state_a_faster_reply_with_its_percentage_when_the_ranges_do_not_overlap() {
    let previous = ledger(
        "0.2.134",
        vec![measurement("Model-A", "code_long_prompt", 55.0)],
    );
    let current = ledger(
        "0.2.135",
        vec![measurement("Model-A", "code_long_prompt", 65.0)],
    );

    let section_text = release_speed_section(&previous, &current);

    assert!(section_text.starts_with(
        "## Speed\n\n> 0.2.135 against 0.2.134, measured on a Mac with an M5 Pro and 48 GB."
    ));
    assert!(
        section_text
            .contains("Replies of 500 tokens, code at temperature 0.7, 3 interleaved rounds")
    );
    assert_eq!(
        row_for(&section_text, "Model-A"),
        "| Model-A, 16 GB limit, streamed from the SSD, writing code after a 10,000-token prompt, DFlash drafter | 2,400 (unchanged) | 4.2 s (unchanged) | 55 → 65 (+18%) | – |"
    );
}

#[test]
fn should_state_a_slower_reply_the_same_way_as_a_faster_one() {
    let previous = ledger(
        "0.2.134",
        vec![measurement("Model-A", "essay_long_prompt", 65.0)],
    );
    let current = ledger(
        "0.2.135",
        vec![measurement("Model-A", "essay_long_prompt", 55.0)],
    );

    let section_text = release_speed_section(&previous, &current);

    assert!(row_for(&section_text, "Model-A").ends_with("| 65 → 55 (-15%) | – |"));
}

#[test]
fn should_call_a_difference_inside_the_spread_unchanged() {
    let previous = ledger(
        "0.2.134",
        vec![measurement("Model-A", "code_long_prompt", 55.0)],
    );
    let current = ledger(
        "0.2.135",
        vec![measurement("Model-A", "code_long_prompt", 57.0)],
    );

    let section_text = release_speed_section(&previous, &current);

    assert!(row_for(&section_text, "Model-A").ends_with("| 57 (unchanged) | – |"));
}

#[test]
fn should_print_no_percentages_when_the_macs_differ() {
    let previous = ledger(
        "0.2.134",
        vec![measurement("Model-A", "code_long_prompt", 55.0)],
    );
    let mut current = ledger(
        "0.2.135",
        vec![measurement("Model-A", "code_long_prompt", 65.0)],
    );
    current.machine.chip = "M4".to_owned();

    let section_text = release_speed_section(&previous, &current);

    assert!(section_text.contains(
        "The previous figures were taken on a Mac with an M5 Pro and 48 GB, so no change is stated."
    ));
    assert!(row_for(&section_text, "Model-A").ends_with("| 55 → 65 | – |"));
}

#[test]
fn should_label_new_workloads_and_list_ones_no_longer_measured() {
    let previous = ledger(
        "0.2.134",
        vec![measurement("Model-Old", "code_long_prompt", 30.0)],
    );
    let current = ledger(
        "0.2.135",
        vec![measurement("Model-New", "code_long_prompt", 40.0)],
    );

    let section_text = release_speed_section(&previous, &current);

    assert!(row_for(&section_text, "Model-New").ends_with("| 40 (new) | – |"));
    assert!(section_text.contains(
        "Not measured in this release: Model-Old, 16 GB limit, writing code after a 10,000-token prompt."
    ));
}

#[test]
fn should_not_compare_a_repackaged_model_or_another_memory_limit() {
    let previous = ledger(
        "0.2.134",
        vec![measurement("Model-A", "code_long_prompt", 30.0)],
    );
    let mut repackaged = measurement("Model-A", "code_long_prompt", 40.0);
    repackaged.model_revision = Some("revision-b".to_owned());
    let mut other_limit = measurement("Model-A", "code_long_prompt", 40.0);
    other_limit.maximum_mlx_memory_gb = None;
    let current = ledger("0.2.135", vec![repackaged, other_limit]);

    let section_text = release_speed_section(&previous, &current);

    let new_row_count = section_text
        .lines()
        .filter(|line| line.ends_with("(new) | – |"))
        .count();
    assert_eq!(new_row_count, 2, "neither row is compared with the old one");
    assert!(
        section_text.contains("Not measured in this release: Model-A, 16 GB limit"),
        "the old row has no match"
    );
    assert!(section_text.contains("Model-A, this Mac's own memory limit, streamed from the SSD"));
}

#[test]
fn should_say_when_a_model_starts_or_stops_drafting() {
    let mut plain_before = measurement("Model-Starts", "code_long_prompt", 20.0);
    plain_before.drafter = DrafterInUse::None;
    let mut drafting_before = measurement("Model-Stops", "code_long_prompt", 20.0);
    drafting_before.drafter = DrafterInUse::Dflash2;
    let mut drafting_now = measurement("Model-Starts", "code_long_prompt", 40.0);
    drafting_now.drafter = DrafterInUse::Dflash2;
    let mut plain_now = measurement("Model-Stops", "code_long_prompt", 20.0);
    plain_now.drafter = DrafterInUse::None;
    let previous = ledger("0.2.134", vec![plain_before, drafting_before]);
    let current = ledger("0.2.135", vec![drafting_now, plain_now]);

    let section_text = release_speed_section(&previous, &current);

    assert!(row_for(&section_text, "Model-Starts").contains("prompt, now drafting with DFlash2 |"));
    assert!(row_for(&section_text, "Model-Stops").contains("prompt, no longer drafting |"));
}

#[test]
fn should_show_no_prompt_reading_speed_for_a_short_prompt() {
    let mut short_prompt = measurement("Model-A", "essay_short_prompt", 60.0);
    short_prompt.figures.prompt_tokens = 42;
    short_prompt.figures.seconds_to_first_token = spread(0.31, 0.3, 0.33);
    let previous = ledger("0.2.134", vec![short_prompt.clone()]);
    let current = ledger("0.2.135", vec![short_prompt]);

    let section_text = release_speed_section(&previous, &current);

    assert!(row_for(&section_text, "Model-A").contains(
        "writing an essay after a short prompt, DFlash drafter | – | 0.3 s (unchanged) |"
    ));
}

fn measurement_with_peak(
    reply_median: f64,
    peak_gb: f64,
) -> crate::release_speed::speed_ledger::SpeedMeasurement {
    const BYTES_PER_GB: f64 = 1_073_741_824.0;
    let mut measured = measurement("Model-A", "code_long_prompt", reply_median);
    measured.figures.peak_mlx_memory_bytes = Some(spread(
        peak_gb * BYTES_PER_GB,
        (peak_gb - 0.1) * BYTES_PER_GB,
        (peak_gb + 0.1) * BYTES_PER_GB,
    ));
    measured
}

#[test]
fn should_state_a_change_in_peak_memory_in_gb_with_its_percentage() {
    let previous = ledger("0.2.134", vec![measurement_with_peak(55.0, 16.0)]);
    let current = ledger("0.2.135", vec![measurement_with_peak(55.0, 12.0)]);

    let section_text = release_speed_section(&previous, &current);

    assert!(
        row_for(&section_text, "Model-A").ends_with("| 16.0 GB → 12.0 GB (-25%) |"),
        "{section_text}"
    );
}

#[test]
fn should_call_peak_memory_inside_the_spread_unchanged() {
    let previous = ledger("0.2.134", vec![measurement_with_peak(55.0, 16.0)]);
    let current = ledger("0.2.135", vec![measurement_with_peak(55.0, 16.05)]);

    let section_text = release_speed_section(&previous, &current);

    assert!(
        row_for(&section_text, "Model-A").ends_with("GB (unchanged) |"),
        "{section_text}"
    );
}

#[test]
fn should_print_peak_memory_alone_when_the_previous_release_did_not_measure_it() {
    // Edge: the row itself is not new, so the figure must not read "(new)".
    let previous = ledger(
        "0.2.134",
        vec![measurement("Model-A", "code_long_prompt", 55.0)],
    );
    let current = ledger("0.2.135", vec![measurement_with_peak(55.0, 12.0)]);

    let section_text = release_speed_section(&previous, &current);

    assert!(
        row_for(&section_text, "Model-A").ends_with("| 12.0 GB |"),
        "{section_text}"
    );
}

#[test]
fn should_not_compare_long_prompts_of_different_lengths() {
    let previous = ledger(
        "0.2.134",
        vec![measurement("Model-A", "code_long_prompt", 40.0)],
    );
    let mut halved_prompt = measurement("Model-A", "code_long_prompt", 60.0);
    halved_prompt.figures.prompt_tokens = 5_096;
    let current = ledger("0.2.135", vec![halved_prompt]);

    let section_text = release_speed_section(&previous, &current);

    let halved_row = row_for(&section_text, "Model-A");
    assert!(halved_row.contains("after a 5,000-token prompt"));
    assert!(halved_row.ends_with("| 60 (new) | – |"), "{halved_row}");
    assert!(section_text.contains(
        "Not measured in this release: Model-A, 16 GB limit, writing code after a 10,000-token prompt."
    ));
}

#[test]
fn should_compare_long_prompts_a_tenth_apart_and_not_further() {
    let previous = ledger(
        "0.2.134",
        vec![measurement("Model-A", "code_long_prompt", 40.0)],
    );
    let mut a_tenth_shorter = measurement("Model-A", "code_long_prompt", 60.0);
    a_tenth_shorter.figures.prompt_tokens = 9_108;
    let mut just_past_a_tenth = measurement("Model-A", "code_long_prompt", 60.0);
    just_past_a_tenth.figures.prompt_tokens = 9_107;

    let compared_text = release_speed_section(&previous, &ledger("0.2.135", vec![a_tenth_shorter]));
    let separate_text =
        release_speed_section(&previous, &ledger("0.2.135", vec![just_past_a_tenth]));

    assert!(row_for(&compared_text, "Model-A").ends_with("| 40 → 60 (+50%) | – |"));
    assert!(row_for(&separate_text, "Model-A").ends_with("| 60 (new) | – |"));
}

#[test]
fn should_never_compare_a_short_prompt_with_a_long_one() {
    let previous = ledger(
        "0.2.134",
        vec![measurement("Model-A", "code_long_prompt", 40.0)],
    );
    let mut short_prompt = measurement("Model-A", "code_long_prompt", 60.0);
    short_prompt.figures.prompt_tokens = 499;

    let section_text = release_speed_section(&previous, &ledger("0.2.135", vec![short_prompt]));

    assert!(row_for(&section_text, "Model-A").ends_with("| 60 (new) | – |"));
}

#[test]
fn should_say_one_reply_a_task_when_each_task_was_measured_once() {
    let previous = ledger(
        "0.2.134",
        vec![measurement("Model-A", "code_long_prompt", 55.0)],
    );
    let mut measured_once = measurement("Model-A", "code_long_prompt", 65.0);
    measured_once.figures.round_count = 1;
    let current = ledger("0.2.135", vec![measured_once]);

    let section_text = release_speed_section(&previous, &current);

    assert!(section_text.contains("code at temperature 0.7, one reply a task. A change"));
    assert!(!section_text.contains("median shown"));
}
