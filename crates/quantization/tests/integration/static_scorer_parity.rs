//! Structural priorities must match the Python values to float rounding.

use astronomical_quantization::sensitivity::static_scorer::structural_priority;

use super::fixtures;

#[test]
#[ignore = "pending implementation of structural_priority"]
fn should_reproduce_the_python_structural_priorities() {
    let fixture = fixtures::static_scorer_parity();

    for (module_name, expected_priority) in &fixture.structural_priority {
        let priority = structural_priority(module_name, fixture.n_layers);
        assert!(
            (priority - expected_priority).abs() < 1e-9,
            "priority differs for {module_name}: {priority} vs {expected_priority}"
        );
    }
}
