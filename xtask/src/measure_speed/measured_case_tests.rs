use super::MeasuredCase;
use crate::measure_speed::served_conditions::ServedConditions;
use crate::measure_speed::speed_prompts::SpeedTask;
use crate::measure_speed::speed_rounds::ModelSpeedRounds;
use crate::measure_speed::stream_timing::ReplyMeasurement;
use crate::release_speed::speed_ledger::{DrafterInUse, WeightsPlacement};

fn measured_case(
    temperature_override: Option<f64>,
    speed_rounds: ModelSpeedRounds,
) -> MeasuredCase {
    MeasuredCase {
        model_id: "Model-A".to_owned(),
        served_conditions: ServedConditions {
            maximum_mlx_memory_gb: Some(16),
            mlx_memory_limit_bytes: 16 * 1_073_741_824,
            weights_placement: WeightsPlacement::Streamed,
            drafter: DrafterInUse::Dflash,
            multi_token_prediction_draft_tokens: Some(1),
        },
        speed_rounds,
        temperature_override,
        reply_tokens_requested: 500,
        long_prompt_tokens: 10_000,
    }
}

fn rounds_of(speed_tasks: &[SpeedTask]) -> ModelSpeedRounds {
    let mut speed_rounds = ModelSpeedRounds::default();
    for speed_task in speed_tasks {
        for reply_rate in [60.0, 62.0, 64.0] {
            speed_rounds.record(
                *speed_task,
                ReplyMeasurement {
                    prompt_tokens: 10_000,
                    completion_tokens: 500,
                    seconds_to_first_token: 4.0,
                    reply_tokens_per_second: Some(reply_rate),
                },
            );
        }
    }
    speed_rounds
}

#[test]
fn should_record_each_task_with_its_own_temperature_and_the_served_conditions() {
    let case = measured_case(None, rounds_of(&SpeedTask::ALL));

    let ledger_measurements = case.ledger_measurements(Some("revision-a"));

    assert_eq!(ledger_measurements.len(), 4);
    let code_measurement = ledger_measurements
        .iter()
        .find(|ledger_measurement| ledger_measurement.task == "code_long_prompt")
        .expect("the long code task");
    assert_eq!(code_measurement.temperature, 0.7);
    assert_eq!(
        code_measurement.model_revision.as_deref(),
        Some("revision-a")
    );
    assert_eq!(code_measurement.maximum_mlx_memory_gb, Some(16));
    assert_eq!(code_measurement.drafter, DrafterInUse::Dflash);
    assert_eq!(
        code_measurement.figures.reply_tokens_per_second.median,
        62.0
    );
    let essay_measurement = ledger_measurements
        .iter()
        .find(|ledger_measurement| ledger_measurement.task == "essay_short_prompt")
        .expect("the short essay task");
    assert_eq!(essay_measurement.temperature, 1.0);
}

#[test]
fn should_leave_out_tasks_without_their_rounds_and_apply_a_temperature_override() {
    let case = measured_case(Some(0.6), rounds_of(&[SpeedTask::LongPromptEssay]));

    let ledger_measurements = case.ledger_measurements(None);

    assert_eq!(ledger_measurements.len(), 1);
    assert_eq!(ledger_measurements[0].task, "essay_long_prompt");
    assert_eq!(ledger_measurements[0].temperature, 0.6);
    assert_eq!(ledger_measurements[0].model_revision, None);
}

#[test]
fn should_record_nothing_for_a_case_with_no_rounds() {
    assert!(
        measured_case(None, ModelSpeedRounds::default())
            .ledger_measurements(None)
            .is_empty()
    );
}
