use super::{NeuralEnginePrefillAdmission, NeuralEnginePrefillMemoryRequirements};

const GIBIBYTE: u64 = 1024 * 1024 * 1024;

fn requirements(active_memory_ceiling_bytes: u64) -> NeuralEnginePrefillMemoryRequirements {
    NeuralEnginePrefillMemoryRequirements {
        resident_model_payload_bytes: 5 * GIBIBYTE,
        maximum_context_state_bytes: 4 * GIBIBYTE,
        fp16_program_weight_bytes: 10 * GIBIBYTE,
        active_memory_ceiling_bytes,
    }
}

#[test]
fn should_admit_programs_that_fit_beside_everything_mlx_can_grow_to() {
    assert_eq!(
        requirements(40 * GIBIBYTE).decide(),
        NeuralEnginePrefillAdmission::Admitted {
            estimated_neural_engine_bytes: 18 * GIBIBYTE,
        }
    );
}

#[test]
fn should_admit_at_exactly_the_ceiling() {
    let exact_ceiling = (5 + 4 + 2 + 18) * GIBIBYTE;

    assert!(matches!(
        requirements(exact_ceiling).decide(),
        NeuralEnginePrefillAdmission::Admitted { .. }
    ));
}

#[test]
fn should_refuse_one_byte_below_the_ceiling() {
    let exact_ceiling = (5 + 4 + 2 + 18) * GIBIBYTE;

    assert_eq!(
        requirements(exact_ceiling - 1).decide(),
        NeuralEnginePrefillAdmission::Refused {
            estimated_neural_engine_bytes: 18 * GIBIBYTE,
            required_bytes: exact_ceiling,
            active_memory_ceiling_bytes: exact_ceiling - 1,
        }
    );
}

#[test]
fn should_refuse_instead_of_wrapping_when_the_inputs_overflow() {
    let overflowing = NeuralEnginePrefillMemoryRequirements {
        resident_model_payload_bytes: u64::MAX,
        maximum_context_state_bytes: u64::MAX,
        fp16_program_weight_bytes: u64::MAX,
        active_memory_ceiling_bytes: u64::MAX - 1,
    };

    assert!(matches!(
        overflowing.decide(),
        NeuralEnginePrefillAdmission::Refused { .. }
    ));
}
