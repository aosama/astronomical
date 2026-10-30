//! Unit contracts for the pure pieces of the fused boundary-checkpoint
//! dispatch: template-argument assembly and kernel-output unpacking. The
//! launch itself and the boundary slicing need live GPU work; these tests pin
//! the facts every launch depends on before Metal is ever submitted.

use super::gated_delta_boundary_checkpoints::{
    checkpoint_template_arguments, unpack_checkpoint_kernel_outputs,
};
use super::gated_delta_sequence_contract::{GatedDeltaSequenceShape, template_arguments};
use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime, MlxRuntimeError};
use mlx_c_rust::{MlxArray, MlxDtype, MlxMetalKernelTemplateArgument};
use serial_test::serial;

fn test_sequence_shape() -> GatedDeltaSequenceShape {
    GatedDeltaSequenceShape {
        batch_size: 2,
        token_count: 7,
        key_head_count: 4,
        value_head_count: 8,
        key_head_dimension: 128,
        value_head_dimension: 32,
    }
}

fn test_runtime() -> MlxRuntime {
    MlxRuntime::initialize(
        MlxMemoryLimits::new(512 * 1024 * 1024, 512 * 1024 * 1024)
            .expect("the checkpoint unpack test memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize")
}

/// Stands in for one kernel output; the width marks its launch position so a
/// swapped unpack order cannot pass by accident.
fn kernel_output(width: i32) -> MlxArray {
    let elements: Vec<i32> = (0..width).collect();
    MlxArray::from_i32(&elements, &[1, width]).expect("the kernel output fixture should upload")
}

#[test]
fn should_extend_shape_invariant_template_arguments_with_one_batch_integer() {
    let extended =
        checkpoint_template_arguments(test_sequence_shape(), MlxDtype::Float16, MlxDtype::Float32);
    let shape_invariant =
        template_arguments(test_sequence_shape(), MlxDtype::Float16, MlxDtype::Float32);
    assert_eq!(extended.len(), shape_invariant.len() + 1);
    assert_eq!(extended[..shape_invariant.len()], shape_invariant[..]);
    assert_eq!(
        extended.last(),
        Some(&MlxMetalKernelTemplateArgument::Integer {
            name: "B",
            integer_template_argument: 2,
        })
    );
}

#[test]
fn should_forward_input_and_state_dtypes_ahead_of_the_batch_integer() {
    let extended =
        checkpoint_template_arguments(test_sequence_shape(), MlxDtype::BFloat16, MlxDtype::Float32);
    assert_eq!(
        extended.first(),
        Some(&MlxMetalKernelTemplateArgument::Dtype {
            name: "InT",
            dtype: MlxDtype::BFloat16,
        })
    );
    assert_eq!(
        extended.get(1),
        Some(&MlxMetalKernelTemplateArgument::Dtype {
            name: "StT",
            dtype: MlxDtype::Float32,
        })
    );
}

#[test]
fn should_carry_each_batch_size_into_the_b_integer() {
    // The checkpoint-write source indexes boundary states with `B`, so the
    // smallest valid batch and a wide batch must both bake through as-is.
    let minimum = checkpoint_template_arguments(
        GatedDeltaSequenceShape {
            batch_size: 1,
            ..test_sequence_shape()
        },
        MlxDtype::Float16,
        MlxDtype::Float32,
    );
    let wide = checkpoint_template_arguments(
        GatedDeltaSequenceShape {
            batch_size: i32::MAX,
            ..test_sequence_shape()
        },
        MlxDtype::Float16,
        MlxDtype::Float32,
    );
    assert_eq!(
        minimum.last(),
        Some(&MlxMetalKernelTemplateArgument::Integer {
            name: "B",
            integer_template_argument: 1,
        })
    );
    assert_eq!(
        wide.last(),
        Some(&MlxMetalKernelTemplateArgument::Integer {
            name: "B",
            integer_template_argument: i32::MAX,
        })
    );
}

#[test]
#[serial]
fn should_unpack_three_kernel_outputs_in_launch_order() {
    // The runtime installs the MLX allocator that backs `MlxArray::from_i32`,
    // matching every other direct-MLX unit test in this crate.
    let _runtime = test_runtime();
    let kernel_outputs = unpack_checkpoint_kernel_outputs(vec![
        kernel_output(1),
        kernel_output(2),
        kernel_output(3),
    ])
    .expect("a three-output launch should unpack");
    assert_eq!(kernel_outputs.sequence_outputs.shape(), [1, 1]);
    assert_eq!(kernel_outputs.next_recurrent_state.shape(), [1, 2]);
    assert_eq!(
        kernel_outputs.packed_recurrent_boundary_states.shape(),
        [1, 3]
    );
}

#[test]
#[serial]
fn should_reject_a_launch_without_sequence_outputs() {
    // The runtime installs the MLX allocator that backs `MlxArray::from_i32`,
    // matching every other direct-MLX unit test in this crate.
    let _runtime = test_runtime();
    let error =
        unpack_checkpoint_kernel_outputs(Vec::new()).expect_err("an empty launch must not unpack");
    let MlxRuntimeError::RuntimeOperation { description, .. } = error else {
        panic!("an unpack failure should be a runtime operation error");
    };
    assert_eq!(
        description,
        "checkpoint kernel did not return sequence outputs"
    );
}

#[test]
#[serial]
fn should_reject_a_launch_missing_the_final_recurrent_state() {
    // The runtime installs the MLX allocator that backs `MlxArray::from_i32`,
    // matching every other direct-MLX unit test in this crate.
    let _runtime = test_runtime();
    let error = unpack_checkpoint_kernel_outputs(vec![kernel_output(1)])
        .expect_err("a one-output launch must not unpack");
    let MlxRuntimeError::RuntimeOperation { description, .. } = error else {
        panic!("an unpack failure should be a runtime operation error");
    };
    assert_eq!(
        description,
        "checkpoint kernel did not return final recurrent state"
    );
}

#[test]
#[serial]
fn should_reject_a_launch_missing_only_the_boundary_states() {
    // The runtime installs the MLX allocator that backs `MlxArray::from_i32`,
    // matching every other direct-MLX unit test in this crate.
    let _runtime = test_runtime();
    // Boundary condition: the sequence outputs and final state are present
    // and only the third array is missing, which must still be rejected
    // before any boundary slicing can index a missing tensor.
    let error = unpack_checkpoint_kernel_outputs(vec![kernel_output(1), kernel_output(2)])
        .expect_err("a two-output launch must not unpack");
    let MlxRuntimeError::RuntimeOperation { description, .. } = error else {
        panic!("an unpack failure should be a runtime operation error");
    };
    assert_eq!(
        description,
        "checkpoint kernel did not return boundary recurrent states"
    );
}
