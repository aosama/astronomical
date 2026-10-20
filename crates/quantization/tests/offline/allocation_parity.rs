//! The allocator must reproduce the Python reference exactly on the
//! synthetic table. A deliberate policy change in the port shows up here as
//! a failing case to be re-pinned, never as silent drift.

use std::collections::BTreeMap;

use astronomical_quantization::allocation::{AllocationConstraints, optimize_mixed_precision};
use astronomical_quantization::bit_width::{BitWidth, CandidateBitWidths, GroupSize};
use astronomical_quantization::sensitivity::{
    SensitivityMethod, SensitivityResult, SensitivityTable,
};

use super::fixtures::{self, FixtureAllocationCase};

fn fixture_table() -> SensitivityTable {
    let fixture = fixtures::allocation_parity();
    SensitivityTable {
        method: SensitivityMethod::ExactBfloat16Reference,
        results: fixture
            .sensitivity_table
            .iter()
            .map(|entry| SensitivityResult {
                module_name: entry.layer_name.clone(),
                kl_divergence_by_bit_width: entry
                    .sensitivities
                    .iter()
                    .map(|(bits_text, value)| {
                        let bits: u32 = bits_text.parse().expect("fixture bit keys are integers");
                        (
                            BitWidth::new(bits).expect("fixture bit widths are valid"),
                            fixtures::fixture_kl_value(value),
                        )
                    })
                    .collect(),
                logical_parameter_count: entry.logical_parameter_count,
            })
            .collect(),
    }
}

fn constraints_for(case: &FixtureAllocationCase) -> AllocationConstraints {
    AllocationConstraints {
        target_bits_per_weight: case.constraints.target_bits_per_weight,
        candidate_bit_widths: Some(
            CandidateBitWidths::new(&case.constraints.candidate_bits)
                .expect("fixture ladders are valid"),
        ),
        group_size: GroupSize::new(64).expect("valid"),
        protect_boundary_blocks: case.constraints.protect_boundary_blocks,
        boundary_block_count: 1,
        floor_components_per_block: case.constraints.floor_components_per_block,
        maximum_low_bit_block_run: case.constraints.maximum_low_bit_block_run,
    }
}

#[test]
fn should_reproduce_the_python_allocation_for_every_fixture_case() {
    let table = fixture_table();

    for case in &fixtures::allocation_parity().cases {
        let allocation = optimize_mixed_precision(&table, &constraints_for(case))
            .expect("the fixture case should allocate");

        let rust_bits: BTreeMap<String, u32> = allocation
            .decisions
            .iter()
            .map(|decision| (decision.module_name.clone(), decision.bit_width.bits()))
            .collect();
        assert_eq!(
            rust_bits, case.allocation,
            "allocation differs for target {} with ladder {:?}",
            case.constraints.target_bits_per_weight, case.constraints.candidate_bits
        );
        assert!((allocation.achieved_bits_per_weight - case.achieved_bits_per_weight).abs() < 1e-9);
        assert_eq!(
            allocation.module_count_at_highest_width,
            case.module_count_at_highest_width
        );
        assert_eq!(
            allocation.module_count_at_lowest_width,
            case.module_count_at_lowest_width
        );
    }
}
