//! Unit tests for the sensitivity report renderer.

use std::collections::BTreeMap;

use super::*;
use crate::bit_width::BitWidth;
use crate::sensitivity::{SensitivityMethod, SensitivityResult};

#[test]
fn should_say_so_when_there_are_no_modules_to_report() {
    let empty_table = SensitivityTable {
        method: SensitivityMethod::ExactBfloat16Reference,
        results: Vec::new(),
    };

    let report = render_sensitivity_report(&empty_table, 20);

    assert_eq!(report, "  (no modules to report)\n");
}

#[test]
#[ignore = "pending implementation of render_sensitivity_report"]
fn should_rank_modules_by_descending_divergence_per_width_and_cut_at_top_n() {
    let table = SensitivityTable {
        method: SensitivityMethod::ExactBfloat16Reference,
        results: vec![
            SensitivityResult {
                module_name: "model.layers.0.mlp.up_proj".to_owned(),
                kl_divergence_by_bit_width: BTreeMap::from([(
                    BitWidth::new(4).expect("valid"),
                    0.1,
                )]),
                logical_parameter_count: 10,
            },
            SensitivityResult {
                module_name: "model.layers.1.mlp.up_proj".to_owned(),
                kl_divergence_by_bit_width: BTreeMap::from([(
                    BitWidth::new(4).expect("valid"),
                    0.9,
                )]),
                logical_parameter_count: 10,
            },
        ],
    };

    let report = render_sensitivity_report(&table, 1);

    assert!(report.contains("most-sensitive modules at 4-bit"));
    assert!(report.contains("model.layers.1.mlp.up_proj"));
    assert!(!report.contains("model.layers.0.mlp.up_proj"));
}
