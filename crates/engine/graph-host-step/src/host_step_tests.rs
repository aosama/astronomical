//! A host step sees what the GPU computed before it, and the GPU after it
//! sees what the step wrote, inside one evaluation.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread::ThreadId;

use mlx_bridge::{MlxArray, MlxDtype};
use runtime_integration::{MlxMemoryLimits, MlxRuntime};
use serial_test::file_serial;

use super::{
    HostStep, HostStepFailed, HostStepInputs, HostStepOutput, append_host_step, set_fast_synch,
};

const ELEMENT_COUNT: usize = 4_096;
const BYTES_PER_ELEMENT: usize = std::mem::size_of::<i32>();

fn test_runtime() -> MlxRuntime {
    MlxRuntime::initialize(
        MlxMemoryLimits::new(1024 * 1024 * 1024, 256 * 1024 * 1024)
            .expect("the host step test memory limits are valid"),
    )
    .expect("the direct MLX runtime initializes")
}

/// Runs `test_body` under each of MLX's two orderings of a host step among
/// GPU steps: its default, Metal shared events, and then its fast synch.
fn in_each_ordering(test_name: &str, test_body: fn(&MlxRuntime)) {
    if skip_without_metal(test_name) {
        return;
    }
    let runtime = test_runtime();
    for is_fast_synch in [false, true] {
        set_fast_synch(is_fast_synch);
        test_body(&runtime);
    }
    set_fast_synch(false);
}

fn skip_without_metal(test_name: &str) -> bool {
    test_harness::skip_without(test_harness::PlatformRequirement::MetalGpu, test_name)
}

/// What the test thread can see of a step that runs on another thread.
#[derive(Default)]
struct StepRecord {
    run_count: AtomicUsize,
    drop_count: AtomicUsize,
    ran_on_the_building_thread: AtomicBool,
}

/// Writes three times each element of its first input, plus one, wrapping
/// as the GPU's integers do. The GPU cannot have produced that by itself, and
/// the step cannot produce it without the GPU's values.
struct TripleAndAddOne {
    record: Arc<StepRecord>,
    building_thread: ThreadId,
}

impl TripleAndAddOne {
    fn boxed(record: &Arc<StepRecord>) -> Box<dyn HostStep> {
        Box::new(Self {
            record: Arc::clone(record),
            building_thread: std::thread::current().id(),
        })
    }
}

impl HostStep for TripleAndAddOne {
    fn run(&mut self, inputs: HostStepInputs<'_>, output: &mut [u8]) -> Result<(), HostStepFailed> {
        self.record.run_count.fetch_add(1, Ordering::SeqCst);
        if std::thread::current().id() == self.building_thread {
            self.record
                .ran_on_the_building_thread
                .store(true, Ordering::SeqCst);
        }
        let input_bytes = inputs.bytes(0).ok_or(HostStepFailed)?;
        if input_bytes.len() != output.len() {
            return Err(HostStepFailed);
        }
        let (input_elements, _) = input_bytes.as_chunks::<BYTES_PER_ELEMENT>();
        let (output_elements, _) = output.as_chunks_mut::<BYTES_PER_ELEMENT>();
        for (input_element, output_element) in input_elements.iter().zip(output_elements) {
            let gpu_value = i32::from_ne_bytes(*input_element);
            *output_element = gpu_value.wrapping_mul(3).wrapping_add(1).to_ne_bytes();
        }
        Ok(())
    }
}

impl Drop for TripleAndAddOne {
    fn drop(&mut self) {
        self.record.drop_count.fetch_add(1, Ordering::SeqCst);
    }
}

/// Reports a failure without writing anything.
struct AlwaysFails;

impl HostStep for AlwaysFails {
    fn run(&mut self, _: HostStepInputs<'_>, _: &mut [u8]) -> Result<(), HostStepFailed> {
        Err(HostStepFailed)
    }
}

/// Panics, as a step with a defect would.
struct AlwaysPanics;

impl HostStep for AlwaysPanics {
    fn run(&mut self, _: HostStepInputs<'_>, _: &mut [u8]) -> Result<(), HostStepFailed> {
        panic!("a host step with a defect");
    }
}

fn ascending_values(first_value: i32, element_count: usize) -> Vec<i32> {
    (0..element_count)
        .map(|element_index| first_value + element_index as i32)
        .collect()
}

/// `gpu_input` + 1 on the GPU, then the step, then doubled on the GPU.
/// Nothing is evaluated here, so `gpu_input` may itself be lazy.
fn gpu_step_gpu(
    runtime: &MlxRuntime,
    gpu_input: &MlxArray,
    element_count: usize,
    record: &Arc<StepRecord>,
) -> MlxArray {
    let one = MlxArray::from_i32(&[1], &[1]).expect("one");
    let two = MlxArray::from_i32(&[2], &[1]).expect("two");
    let before_step = runtime.add(gpu_input, &one).expect("the GPU step before");
    let step_output = append_host_step(
        &[&before_step],
        HostStepOutput {
            shape: &[element_count as i32],
            dtype: MlxDtype::Int32,
        },
        TripleAndAddOne::boxed(record),
    )
    .expect("the step joins the graph");
    runtime
        .multiply(&step_output, &two)
        .expect("the GPU step after")
}

#[test]
#[file_serial]
fn should_run_between_the_gpu_steps_it_reads_and_the_gpu_steps_that_read_it() {
    in_each_ordering(
        "should_run_between_the_gpu_steps_it_reads_and_the_gpu_steps_that_read_it",
        gpu_step_gpu_rounds,
    );
}

fn gpu_step_gpu_rounds(runtime: &MlxRuntime) {
    let record = Arc::new(StepRecord::default());
    // Many rounds with different values: a step that ran before the GPU had
    // written its input, or a GPU step that read the output early, would
    // show the previous round's values or none.
    for round in 0..200 {
        let host_values = ascending_values(round * 7, ELEMENT_COUNT);
        let gpu_input = MlxArray::from_i32(&host_values, &[ELEMENT_COUNT as i32]).expect("input");
        let after_step = gpu_step_gpu(runtime, &gpu_input, ELEMENT_COUNT, &record);
        let expected: Vec<i32> = host_values
            .iter()
            .map(|value| ((value + 1) * 3 + 1) * 2)
            .collect();
        assert_eq!(
            after_step.to_vec_i32().expect("the result reads"),
            expected,
            "round {round}"
        );
    }
    assert_eq!(record.run_count.load(Ordering::SeqCst), 200);
    assert_eq!(record.drop_count.load(Ordering::SeqCst), 200);
    assert!(
        !record.ran_on_the_building_thread.load(Ordering::SeqCst),
        "the step runs on MLX's CPU stream thread"
    );
}

#[test]
#[file_serial]
fn should_keep_a_step_a_layer_in_order_within_one_evaluation() {
    in_each_ordering(
        "should_keep_a_step_a_layer_in_order_within_one_evaluation",
        a_step_a_layer_in_one_evaluation,
    );
}

fn a_step_a_layer_in_one_evaluation(runtime: &MlxRuntime) {
    const LAYER_COUNT: usize = 35;
    let record = Arc::new(StepRecord::default());
    // Small values and wrapping arithmetic on both sides, so thirty-five
    // layers of (x + 1) * 3 + 1, doubled, stay comparable.
    let host_values = ascending_values(0, 64);
    let mut layer_output = MlxArray::from_i32(&host_values, &[64]).expect("input");
    for _ in 0..LAYER_COUNT {
        layer_output = gpu_step_gpu(runtime, &layer_output, host_values.len(), &record);
    }
    assert_eq!(
        record.run_count.load(Ordering::SeqCst),
        0,
        "building the graph runs nothing"
    );

    let computed = layer_output.to_vec_i32().expect("one evaluation");

    let expected: Vec<i32> = host_values
        .iter()
        .map(|value| {
            let mut layer_value = *value;
            for _ in 0..LAYER_COUNT {
                layer_value = layer_value
                    .wrapping_add(1)
                    .wrapping_mul(3)
                    .wrapping_add(1)
                    .wrapping_mul(2);
            }
            layer_value
        })
        .collect();
    assert_eq!(computed, expected);
    assert_eq!(record.run_count.load(Ordering::SeqCst), LAYER_COUNT);
    assert_eq!(record.drop_count.load(Ordering::SeqCst), LAYER_COUNT);
}

/// Takes as long as a layer that reads missing experts from storage, and
/// longer, before it writes.
struct SlowTripleAndAddOne {
    inner: TripleAndAddOne,
    delay: std::time::Duration,
}

impl HostStep for SlowTripleAndAddOne {
    fn run(&mut self, inputs: HostStepInputs<'_>, output: &mut [u8]) -> Result<(), HostStepFailed> {
        std::thread::sleep(self.delay);
        self.inner.run(inputs, output)
    }
}

#[test]
#[file_serial]
fn should_hold_the_gpu_steps_after_a_slow_step_until_it_has_written() {
    in_each_ordering(
        "should_hold_the_gpu_steps_after_a_slow_step_until_it_has_written",
        gpu_steps_after_slow_steps,
    );
}

fn gpu_steps_after_slow_steps(runtime: &MlxRuntime) {
    let record = Arc::new(StepRecord::default());
    let two = MlxArray::from_i32(&[2], &[1]).expect("two");
    // From a storage read's millisecond to far longer than any layer waits.
    for delay_milliseconds in [1, 20, 250] {
        let gpu_input = MlxArray::from_i32(&[1, 2, 3, 4], &[4]).expect("input");
        let step_output = append_host_step(
            &[&gpu_input],
            HostStepOutput {
                shape: &[4],
                dtype: MlxDtype::Int32,
            },
            Box::new(SlowTripleAndAddOne {
                inner: TripleAndAddOne {
                    record: Arc::clone(&record),
                    building_thread: std::thread::current().id(),
                },
                delay: std::time::Duration::from_millis(delay_milliseconds),
            }),
        )
        .expect("the step joins the graph");
        let after_step = runtime
            .multiply(&step_output, &two)
            .expect("the GPU step after");
        assert_eq!(
            after_step.to_vec_i32().expect("the result reads"),
            vec![8, 14, 20, 26],
            "a step of {delay_milliseconds} ms"
        );
    }
}

#[test]
#[file_serial]
fn should_read_a_strided_view_in_row_major_order() {
    if skip_without_metal("should_read_a_strided_view_in_row_major_order") {
        return;
    }
    let runtime = test_runtime();
    let record = Arc::new(StepRecord::default());
    let host_values = ascending_values(10, 16);
    let whole = MlxArray::from_i32(&host_values, &[16]).expect("input");
    // Every second element from the third on: a view with an offset and a
    // stride, which is how a layer's routes arrive.
    let every_second = runtime
        .slice(&whole, &[2], &[16], &[2])
        .expect("a strided view");
    let step_output = append_host_step(
        &[&every_second],
        HostStepOutput {
            shape: &[7],
            dtype: MlxDtype::Int32,
        },
        TripleAndAddOne::boxed(&record),
    )
    .expect("the step joins the graph");

    let expected: Vec<i32> = host_values[2..]
        .iter()
        .step_by(2)
        .map(|value| value * 3 + 1)
        .collect();
    assert_eq!(
        step_output.to_vec_i32().expect("the result reads"),
        expected
    );
}

#[test]
#[file_serial]
fn should_fail_the_evaluation_when_the_step_fails_and_serve_the_next_one() {
    in_each_ordering(
        "should_fail_the_evaluation_when_the_step_fails_and_serve_the_next_one",
        failed_steps_then_a_served_one,
    );
}

fn failed_steps_then_a_served_one(runtime: &MlxRuntime) {
    let gpu_input = MlxArray::from_i32(&[1, 2, 3, 4], &[4]).expect("input");
    for failing_step in [
        Box::new(AlwaysFails) as Box<dyn HostStep>,
        Box::new(AlwaysPanics),
    ] {
        let step_output = append_host_step(
            &[&gpu_input],
            HostStepOutput {
                shape: &[4],
                dtype: MlxDtype::Int32,
            },
            failing_step,
        )
        .expect("the step joins the graph");
        assert!(
            step_output.to_vec_i32().is_err(),
            "a failed step fails the evaluation"
        );
    }

    // The CPU stream is not left poisoned: the next evaluation is served.
    let record = Arc::new(StepRecord::default());
    let after_step = gpu_step_gpu(runtime, &gpu_input, 4, &record);
    assert_eq!(
        after_step.to_vec_i32().expect("the result reads"),
        vec![14, 20, 26, 32]
    );
}

#[test]
#[file_serial]
fn should_drop_a_step_once_when_its_output_is_never_evaluated() {
    if skip_without_metal("should_drop_a_step_once_when_its_output_is_never_evaluated") {
        return;
    }
    let _runtime = test_runtime();
    let record = Arc::new(StepRecord::default());
    let gpu_input = MlxArray::from_i32(&[1, 2, 3, 4], &[4]).expect("input");
    let step_output = append_host_step(
        &[&gpu_input],
        HostStepOutput {
            shape: &[4],
            dtype: MlxDtype::Int32,
        },
        TripleAndAddOne::boxed(&record),
    )
    .expect("the step joins the graph");
    assert_eq!(record.drop_count.load(Ordering::SeqCst), 0);

    drop(step_output);

    assert_eq!(record.run_count.load(Ordering::SeqCst), 0);
    assert_eq!(record.drop_count.load(Ordering::SeqCst), 1);
}

#[test]
#[file_serial]
fn should_run_a_step_with_no_inputs_and_an_empty_output() {
    if skip_without_metal("should_run_a_step_with_no_inputs_and_an_empty_output") {
        return;
    }
    let _runtime = test_runtime();
    let record = Arc::new(StepRecord::default());
    let step_output = append_host_step(
        &[],
        HostStepOutput {
            shape: &[0],
            dtype: MlxDtype::Int32,
        },
        Box::new(CountsRunsOnly {
            record: Arc::clone(&record),
        }),
    )
    .expect("the step joins the graph");

    assert!(
        step_output
            .to_vec_i32()
            .expect("the result reads")
            .is_empty()
    );
    assert_eq!(record.run_count.load(Ordering::SeqCst), 1);
}

/// Counts its runs and checks it was handed nothing to read or write.
struct CountsRunsOnly {
    record: Arc<StepRecord>,
}

impl HostStep for CountsRunsOnly {
    fn run(&mut self, inputs: HostStepInputs<'_>, output: &mut [u8]) -> Result<(), HostStepFailed> {
        self.record.run_count.fetch_add(1, Ordering::SeqCst);
        if inputs.count() != 0 || inputs.bytes(0).is_some() || !output.is_empty() {
            return Err(HostStepFailed);
        }
        Ok(())
    }
}

#[test]
#[file_serial]
fn should_refuse_a_negative_output_dimension_and_drop_the_step() {
    if skip_without_metal("should_refuse_a_negative_output_dimension_and_drop_the_step") {
        return;
    }
    let _runtime = test_runtime();
    let record = Arc::new(StepRecord::default());

    let refusal = append_host_step(
        &[],
        HostStepOutput {
            shape: &[4, -1],
            dtype: MlxDtype::Int32,
        },
        TripleAndAddOne::boxed(&record),
    );

    assert!(refusal.is_err());
    assert_eq!(record.drop_count.load(Ordering::SeqCst), 1);
}
