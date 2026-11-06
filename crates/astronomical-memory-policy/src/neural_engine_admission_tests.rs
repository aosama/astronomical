use super::{
    NeuralEnginePrefillAdmission, NeuralEnginePrefillMemoryRequirements, chunk_fits_beside_programs,
};

const GIBIBYTE: u64 = 1024 * 1024 * 1024;
const MODEL_STATE_AND_PROGRAMS: u64 = (5 + 1 + 13) * GIBIBYTE;

fn requirements(active_memory_ceiling_bytes: u64) -> NeuralEnginePrefillMemoryRequirements {
    NeuralEnginePrefillMemoryRequirements {
        resident_model_payload_bytes: 5 * GIBIBYTE,
        one_chunk_state_bytes: GIBIBYTE,
        program_bytes: 13 * GIBIBYTE,
        active_memory_ceiling_bytes,
    }
}

#[test]
fn should_admit_programs_that_fit_beside_the_model() {
    assert_eq!(
        requirements(32 * GIBIBYTE).decide(),
        NeuralEnginePrefillAdmission::Admitted
    );
}

#[test]
fn should_admit_at_exactly_the_ceiling() {
    assert_eq!(
        requirements(MODEL_STATE_AND_PROGRAMS).decide(),
        NeuralEnginePrefillAdmission::Admitted
    );
}

#[test]
fn should_refuse_one_byte_below_the_ceiling() {
    assert_eq!(
        requirements(MODEL_STATE_AND_PROGRAMS - 1).decide(),
        NeuralEnginePrefillAdmission::Refused {
            required_bytes: MODEL_STATE_AND_PROGRAMS,
        }
    );
}

#[test]
fn should_refuse_instead_of_wrapping_when_the_inputs_overflow() {
    let overflowing = NeuralEnginePrefillMemoryRequirements {
        resident_model_payload_bytes: u64::MAX,
        one_chunk_state_bytes: u64::MAX,
        program_bytes: u64::MAX,
        active_memory_ceiling_bytes: u64::MAX - 1,
    };

    assert!(matches!(
        overflowing.decide(),
        NeuralEnginePrefillAdmission::Refused { .. }
    ));
}

#[test]
fn should_let_a_chunk_that_fits_use_the_programs() {
    assert!(chunk_fits_beside_programs(
        10 * GIBIBYTE,
        13 * GIBIBYTE,
        32 * GIBIBYTE
    ));
}

#[test]
fn should_let_a_chunk_reach_exactly_the_ceiling() {
    assert!(chunk_fits_beside_programs(
        19 * GIBIBYTE,
        13 * GIBIBYTE,
        32 * GIBIBYTE
    ));
}

#[test]
fn should_keep_a_chunk_one_byte_past_the_ceiling_on_the_gpu() {
    assert!(!chunk_fits_beside_programs(
        19 * GIBIBYTE + 1,
        13 * GIBIBYTE,
        32 * GIBIBYTE
    ));
}

#[test]
fn should_keep_a_chunk_on_the_gpu_instead_of_wrapping() {
    assert!(!chunk_fits_beside_programs(
        u64::MAX,
        u64::MAX,
        u64::MAX - 1
    ));
}
