//! Tree and chain size sweeps for either drafter generation, on the
//! tree-rounds harness: where a drafter's kept tokens stop paying for the
//! rows a round verifies, at the publisher's recommended temperature (0.6)
//! and at the story's 1.0. Each case prints kept tokens a round, milliseconds a
//! round and decode tok/s beside plain one-token steps.

use serial_test::serial;

use super::dflash2_journey_support::story_cell;
use super::dflash2_kept_per_round::ENGINE_CHILDREN_PER_NODE;
use super::dflash2_tree_oracle::CODE_CELL;
use super::dflash2_tree_rounds::{decode_cells_with_trees, report_plain_decode};

/// Node budgets for the depth sweep, from one draft a round to the trained
/// block's fifteen.
const SWEEP_NODE_BUDGETS: [usize; 6] = [1, 2, 3, 5, 7, 15];
const SWEEP_SEED_COUNT: u64 = 2;

/// The code cell at 0.7, the story at the publisher's recommended
/// temperature for general tasks (0.6, with top-p 0.95 and top-k 20) and
/// the story at 1.0, plain against every tree size and against chains as
/// deep: where each drafter's kept tokens stop paying for the rows a round
/// verifies.
#[test]
#[ignore = "loads a real target and its drafter; run through cargo measure-dflash2-tree-rounds"]
#[serial]
fn should_sweep_tree_sizes_and_chains_for_code_and_story_at_real_temperatures() {
    for base_cell in [
        CODE_CELL,
        story_cell("story 0.6", 600),
        story_cell("story 1.0", 1_000),
    ] {
        report_plain_decode(&base_cell, SWEEP_SEED_COUNT);
        let mut cells = Vec::new();
        for node_budget in SWEEP_NODE_BUDGETS {
            for (shape, children_per_node) in [("tree", ENGINE_CHILDREN_PER_NODE), ("chain", 1)] {
                let mut cell = base_cell;
                cell.label = Box::leak(
                    format!("{}, {node_budget}-node {shape}", base_cell.label).into_boxed_str(),
                );
                cell.node_budget = node_budget;
                cell.children_per_node = children_per_node;
                cells.push(cell);
            }
        }
        decode_cells_with_trees(&cells, SWEEP_SEED_COUNT);
    }
}

/// Tree sizes around the one a mixture-of-experts target serves with (3),
/// the code cell at 0.7 and the story at the publisher's recommended 0.6: whether one to three
/// more nodes keep enough extra tokens to pay for their rows.
#[test]
#[ignore = "loads a real target and its drafter; run through cargo measure-dflash2-tree-rounds"]
#[serial]
fn should_compare_three_to_six_node_trees_at_the_recommended_temperature() {
    for (label, base_cell) in [
        ("code completion 0.7", CODE_CELL),
        ("story 0.6", story_cell("story 0.6", 600)),
    ] {
        let mut cells = Vec::new();
        for node_budget in [3, 4, 5, 6] {
            let mut cell = base_cell;
            cell.label = Box::leak(format!("{label}, {node_budget}-node tree").into_boxed_str());
            cell.node_budget = node_budget;
            cells.push(cell);
        }
        decode_cells_with_trees(&cells, SWEEP_SEED_COUNT);
    }
}
