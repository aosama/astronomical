//! The GPU optimizer against the CPU optimizer it mirrors.

use astronomical_decision_training::anchored_adam::{
    AdamSettings, AnchoredAdam, TRAINING_ADAM_SETTINGS,
};
use serial_test::serial;

use super::super::gpu_test_support::assert_all_close;
use super::*;
use astronomical_model_primitives::synthetic_weights_test_support::initialize_test_runtime;

fn settings() -> AdamSettings {
    AdamSettings {
        learning_rate: 0.05,
        anchor_decay: 0.2,
        maximum_gradient_norm: 1.0,
        ..TRAINING_ADAM_SETTINGS
    }
}

#[test]
#[serial]
fn should_take_the_same_three_steps_as_the_cpu_optimizer() {
    let runtime = initialize_test_runtime();
    let math = GpuMath { runtime: &runtime };
    let anchor_values = [0.5_f32, -0.25, 1.0];
    let mut cpu_parameters = anchor_values.to_vec();
    let mut cpu_optimizer = AnchoredAdam::new(settings(), 3);
    let anchors = vec![math.upload(&anchor_values, &[3]).unwrap()];
    let mut gpu_parameters = vec![math.upload(&anchor_values, &[3]).unwrap()];
    let mut gpu_optimizer = GpuAdam::new(math, settings(), &gpu_parameters).unwrap();
    for gradient_values in [[3.0_f32, 4.0, 0.0], [0.1, -0.2, 0.3], [-0.5, 0.0, 0.25]] {
        let mut cpu_gradient = gradient_values.to_vec();
        cpu_optimizer
            .step(&mut cpu_parameters, &mut cpu_gradient, &anchor_values)
            .unwrap();
        let gradients = vec![math.upload(&gradient_values, &[3]).unwrap()];
        assert_eq!(
            gpu_optimizer
                .step(math, &mut gpu_parameters, &gradients, &anchors)
                .unwrap(),
            StepOutcome::Taken
        );
    }
    assert_all_close(
        &math.download(&gpu_parameters[0]).unwrap(),
        &cpu_parameters,
        "parameters",
    );
}

#[test]
#[serial]
fn should_refuse_a_non_finite_gradient_and_change_nothing() {
    let runtime = initialize_test_runtime();
    let math = GpuMath { runtime: &runtime };
    let anchors = vec![math.upload(&[1.0, 2.0], &[2]).unwrap()];
    let mut parameters = vec![math.upload(&[1.0, 2.0], &[2]).unwrap()];
    let mut optimizer = GpuAdam::new(math, settings(), &parameters).unwrap();
    let poisoned = vec![math.upload(&[f32::NAN, 0.5], &[2]).unwrap()];
    assert_eq!(
        optimizer
            .step(math, &mut parameters, &poisoned, &anchors)
            .unwrap(),
        StepOutcome::RefusedNonFinite
    );
    assert_eq!(math.download(&parameters[0]).unwrap(), vec![1.0, 2.0]);
}
