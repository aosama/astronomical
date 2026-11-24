use std::sync::atomic::AtomicBool;
use std::time::Duration;

use super::{
    FinishedMeasurement, MeasurableProgram, MeasurementAction, MeasurementOutcome,
    ProgramMeasurement, ProgramsToMeasure, WiredMemorySource, action_after_measurement,
    measure_until_settled,
};

const SHORT_LIMIT: Duration = Duration::from_millis(40);
/// Past the settling window, for the journeys that reach a settled reading.
const LONG_LIMIT: Duration = Duration::from_secs(10);
const GIBIBYTE: u64 = 1024 * 1024 * 1024;

/// Sleeps like a real program would, so a settling window passes in a few cycles.
struct FakeProgram {
    runs_before_failing: Option<u32>,
    run_duration: Duration,
}

impl FakeProgram {
    const fn healthy(run_duration: Duration) -> Self {
        Self {
            runs_before_failing: None,
            run_duration,
        }
    }

    const fn failing_after(run_count: u32) -> Self {
        Self {
            runs_before_failing: Some(run_count),
            run_duration: Duration::ZERO,
        }
    }
}

impl MeasurableProgram for FakeProgram {
    fn run_once(&mut self) -> bool {
        std::thread::sleep(self.run_duration);
        match &mut self.runs_before_failing {
            Some(0) => false,
            Some(remaining_runs) => {
                *remaining_runs -= 1;
                true
            }
            None => true,
        }
    }
}

/// Reads `before` once, then `after` from then on.
struct ScriptedWiredMemory {
    next_reading: u64,
    after: u64,
}

impl ScriptedWiredMemory {
    const fn reading(before: u64, after: u64) -> Self {
        Self {
            next_reading: before,
            after,
        }
    }
}

impl WiredMemorySource for ScriptedWiredMemory {
    fn wired_bytes(&mut self) -> u64 {
        std::mem::replace(&mut self.next_reading, self.after)
    }
}

fn programs_of(
    mlp_programs: Vec<FakeProgram>,
    output_gate_programs: Vec<(usize, FakeProgram)>,
) -> ProgramsToMeasure<FakeProgram> {
    ProgramsToMeasure {
        mlp_programs,
        output_gate_programs,
    }
}

fn one_healthy_program(run_duration: Duration) -> ProgramsToMeasure<FakeProgram> {
    programs_of(vec![FakeProgram::healthy(run_duration)], Vec::new())
}

#[test]
fn should_report_a_settled_reading_above_the_weights() {
    let mut programs = one_healthy_program(Duration::from_millis(20));

    let outcome = measure_until_settled(
        &mut programs,
        10 * GIBIBYTE,
        &mut ScriptedWiredMemory::reading(4 * GIBIBYTE, 17 * GIBIBYTE),
        &AtomicBool::new(false),
        LONG_LIMIT,
    );

    assert_eq!(
        outcome,
        MeasurementOutcome::Settled {
            wired_bytes: 13 * GIBIBYTE
        }
    );
}

#[test]
fn should_report_unsettled_when_the_reading_stays_below_the_weights() {
    let mut programs = one_healthy_program(Duration::ZERO);

    let outcome = measure_until_settled(
        &mut programs,
        10 * GIBIBYTE,
        &mut ScriptedWiredMemory::reading(4 * GIBIBYTE, 4 * GIBIBYTE),
        &AtomicBool::new(false),
        SHORT_LIMIT,
    );

    assert_eq!(outcome, MeasurementOutcome::Unsettled);
}

#[test]
fn should_not_read_a_failed_first_query_as_growth() {
    let mut programs = one_healthy_program(Duration::ZERO);

    let outcome = measure_until_settled(
        &mut programs,
        10 * GIBIBYTE,
        &mut ScriptedWiredMemory::reading(0, 30 * GIBIBYTE),
        &AtomicBool::new(false),
        SHORT_LIMIT,
    );

    assert_eq!(outcome, MeasurementOutcome::Unsettled);
}

#[test]
fn should_keep_and_record_a_settled_reading() {
    assert_eq!(
        action_after_measurement(MeasurementOutcome::Settled { wired_bytes: 13 }, false),
        MeasurementAction::Keep { wired_bytes: 13 }
    );
}

#[test]
fn should_never_use_an_unsettled_reading() {
    assert_eq!(
        action_after_measurement(MeasurementOutcome::Unsettled, false),
        MeasurementAction::MeasureAgainAfterUnsettled
    );
}

#[test]
fn should_throw_away_any_reading_the_gpu_ran_through() {
    for outcome in [
        MeasurementOutcome::Settled { wired_bytes: 13 },
        MeasurementOutcome::Unsettled,
        MeasurementOutcome::Cancelled,
    ] {
        assert_eq!(
            action_after_measurement(outcome, true),
            MeasurementAction::MeasureAgain
        );
    }
}

#[test]
fn should_measure_again_after_a_cancel() {
    assert_eq!(
        action_after_measurement(MeasurementOutcome::Cancelled, false),
        MeasurementAction::MeasureAgain
    );
}

#[test]
fn should_give_up_on_a_failed_mlp_program_whatever_the_gpu_did() {
    for has_gpu_been_used in [false, true] {
        assert_eq!(
            action_after_measurement(MeasurementOutcome::MlpProgramFailed, has_gpu_been_used),
            MeasurementAction::GiveUp
        );
    }
}

#[test]
fn should_carry_on_without_gates_after_one_fails() {
    assert_eq!(
        action_after_measurement(MeasurementOutcome::OutputGateFailed, true),
        MeasurementAction::MeasureAgainWithoutOutputGates
    );
}

#[test]
fn should_report_unsettled_when_the_reading_keeps_rising_to_the_limit() {
    let mut programs = one_healthy_program(Duration::ZERO);

    let outcome = measure_until_settled(
        &mut programs,
        GIBIBYTE,
        &mut RisingWiredMemory {
            next_reading: GIBIBYTE,
        },
        &AtomicBool::new(false),
        SHORT_LIMIT,
    );

    assert_eq!(outcome, MeasurementOutcome::Unsettled);
}

#[test]
fn should_not_read_through_a_cancel_that_lands_during_the_runs() {
    let should_cancel = std::sync::Arc::new(AtomicBool::new(false));
    let mut programs = ProgramsToMeasure {
        mlp_programs: vec![CancelsWhileRunning {
            should_cancel: std::sync::Arc::clone(&should_cancel),
        }],
        output_gate_programs: Vec::new(),
    };

    let outcome = measure_until_settled(
        &mut programs,
        GIBIBYTE,
        &mut ScriptedWiredMemory::reading(4 * GIBIBYTE, 17 * GIBIBYTE),
        &should_cancel,
        Duration::ZERO,
    );

    assert_eq!(outcome, MeasurementOutcome::Cancelled);
}

#[test]
fn should_stop_when_cancelled_before_running_anything() {
    let mut programs = programs_of(vec![FakeProgram::failing_after(0)], Vec::new());

    let outcome = measure_until_settled(
        &mut programs,
        10 * GIBIBYTE,
        &mut ScriptedWiredMemory::reading(4 * GIBIBYTE, 17 * GIBIBYTE),
        &AtomicBool::new(true),
        LONG_LIMIT,
    );

    assert_eq!(outcome, MeasurementOutcome::Cancelled);
}

#[test]
fn should_report_an_mlp_program_that_fails() {
    let mut programs = programs_of(vec![FakeProgram::failing_after(2)], Vec::new());

    let outcome = measure_until_settled(
        &mut programs,
        10 * GIBIBYTE,
        &mut ScriptedWiredMemory::reading(4 * GIBIBYTE, 4 * GIBIBYTE),
        &AtomicBool::new(false),
        LONG_LIMIT,
    );

    assert_eq!(outcome, MeasurementOutcome::MlpProgramFailed);
}

#[test]
fn should_tell_an_output_gate_failure_apart_from_an_mlp_failure() {
    let mut programs = programs_of(
        vec![FakeProgram::healthy(Duration::ZERO)],
        vec![(3, FakeProgram::failing_after(1))],
    );

    let outcome = measure_until_settled(
        &mut programs,
        10 * GIBIBYTE,
        &mut ScriptedWiredMemory::reading(4 * GIBIBYTE, 4 * GIBIBYTE),
        &AtomicBool::new(false),
        LONG_LIMIT,
    );

    assert_eq!(outcome, MeasurementOutcome::OutputGateFailed);
}

fn wait_for_finish(
    measurement: &mut ProgramMeasurement<FakeProgram>,
) -> (ProgramsToMeasure<FakeProgram>, MeasurementOutcome) {
    for _attempt in 0..1000 {
        match measurement.take_finished() {
            Some(FinishedMeasurement::Ended(programs, outcome)) => return (programs, outcome),
            Some(FinishedMeasurement::Lost) => panic!("the measuring thread panicked"),
            None => {}
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    panic!("the measurement should end");
}

#[test]
fn should_hand_the_programs_back_after_a_cancel() {
    let mut measurement = ProgramMeasurement::start(
        programs_of(
            vec![FakeProgram::healthy(Duration::from_millis(1))],
            vec![(3, FakeProgram::healthy(Duration::from_millis(1)))],
        ),
        10 * GIBIBYTE,
        ScriptedWiredMemory::reading(4 * GIBIBYTE, 4 * GIBIBYTE),
    )
    .expect("the thread starts");

    measurement.cancel();
    let (programs, outcome) = wait_for_finish(&mut measurement);

    assert_eq!(outcome, MeasurementOutcome::Cancelled);
    assert_eq!(programs.mlp_programs.len(), 1);
    assert_eq!(programs.output_gate_programs.len(), 1);
}

#[test]
fn should_finish_only_once() {
    let mut measurement = ProgramMeasurement::start(
        one_healthy_program(Duration::ZERO),
        10 * GIBIBYTE,
        ScriptedWiredMemory::reading(4 * GIBIBYTE, 4 * GIBIBYTE),
    )
    .expect("the thread starts");
    measurement.cancel();

    wait_for_finish(&mut measurement);

    assert!(measurement.take_finished().is_none());
}

/// Grows on every reading, like a Mac too busy to settle.
struct RisingWiredMemory {
    next_reading: u64,
}

impl WiredMemorySource for RisingWiredMemory {
    fn wired_bytes(&mut self) -> u64 {
        self.next_reading += GIBIBYTE;
        self.next_reading
    }
}

struct CancelsWhileRunning {
    should_cancel: std::sync::Arc<AtomicBool>,
}

impl MeasurableProgram for CancelsWhileRunning {
    fn run_once(&mut self) -> bool {
        self.should_cancel
            .store(true, std::sync::atomic::Ordering::Relaxed);
        true
    }
}

struct PanickingProgram;

impl MeasurableProgram for PanickingProgram {
    fn run_once(&mut self) -> bool {
        panic!("a program that takes its thread down");
    }
}

#[test]
fn should_report_a_measurement_lost_to_a_panic() {
    let mut measurement = ProgramMeasurement::start(
        ProgramsToMeasure {
            mlp_programs: vec![PanickingProgram],
            output_gate_programs: Vec::new(),
        },
        10 * GIBIBYTE,
        ScriptedWiredMemory::reading(4 * GIBIBYTE, 4 * GIBIBYTE),
    )
    .expect("the thread starts");

    for _attempt in 0..1000 {
        match measurement.take_finished() {
            Some(FinishedMeasurement::Lost) => return,
            Some(FinishedMeasurement::Ended(..)) => panic!("the thread should not end cleanly"),
            None => std::thread::sleep(Duration::from_millis(5)),
        }
    }
    panic!("the measurement should end");
}

#[test]
fn should_stop_its_thread_when_dropped() {
    let measurement = ProgramMeasurement::start(
        one_healthy_program(Duration::from_millis(1)),
        10 * GIBIBYTE,
        ScriptedWiredMemory::reading(4 * GIBIBYTE, 4 * GIBIBYTE),
    )
    .expect("the thread starts");

    drop(measurement);
}
