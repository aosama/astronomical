use super::draft_tree::{
    DraftLattice, DraftTree, DraftTreeOptions, KeyedNoise, best_first_draft_tree,
};

/// A lattice of `depth` slots with `candidate_count` candidates whose ids are
/// `100 * slot + candidate`, the given logits, and no edge preference.
fn flat_lattice(depth: usize, candidate_count: usize, unary_scores: Vec<f32>) -> DraftLattice {
    let mut candidate_ids = Vec::new();
    for slot in 0..depth {
        for candidate in 0..candidate_count {
            candidate_ids.push((100 * slot + candidate) as u32);
        }
    }
    DraftLattice::without_edges(depth, candidate_count, candidate_ids, unary_scores)
}

#[test]
fn should_size_a_lattice_without_edges_for_every_parent_and_no_slot() {
    let three_slots = DraftLattice::without_edges(3, 4, vec![7; 12], vec![0.5; 12]);
    assert_eq!(three_slots.anchor_edge_scores, vec![0.0; 4]);
    assert_eq!(three_slots.edge_scores, vec![0.0; 2 * 4 * 4]);
    let no_slots = DraftLattice::without_edges(0, 4, Vec::new(), Vec::new());
    assert!(no_slots.edge_scores.is_empty());
}

fn options(node_budget: usize, children_per_node: usize) -> DraftTreeOptions {
    DraftTreeOptions {
        node_budget,
        children_per_node,
        ..DraftTreeOptions::default()
    }
}

/// Noise that strongly favours one token at one position.
struct FavouredToken {
    position: u64,
    token_id: u32,
}

impl KeyedNoise for FavouredToken {
    fn draw(&self, absolute_position: u64, token_id: u32) -> f64 {
        if absolute_position == self.position && token_id == self.token_id {
            50.0
        } else {
            0.0
        }
    }
}

#[test]
fn should_follow_the_strongest_chain_first_and_hang_each_node_under_its_parent() {
    // Candidate 0 dominates every slot, so the first nodes form its chain.
    let lattice = flat_lattice(3, 3, vec![9.0, 0.0, 0.0, 9.0, 0.0, 0.0, 9.0, 0.0, 0.0]);
    let tree = best_first_draft_tree(&lattice, 10, &options(3, 2), None);
    assert_eq!(
        tree,
        DraftTree {
            token_ids: vec![0, 100, 200],
            parent_indices: vec![None, Some(0), Some(1)],
            depths: vec![0, 1, 2],
        }
    );
}

#[test]
fn should_spend_the_budget_on_siblings_when_a_slot_is_uncertain() {
    // Two equally likely first tokens and a sure second slot: both branches
    // are worth more than going deeper on one of them alone.
    let lattice = flat_lattice(2, 2, vec![0.0, 0.0, 20.0, -20.0]);
    let tree = best_first_draft_tree(&lattice, 0, &options(4, 2), None);
    assert_eq!(tree.token_ids.len(), 4);
    assert_eq!(&tree.parent_indices[..2], &[None, None]);
    assert_eq!(&tree.token_ids[..2], &[0, 1]);
}

#[test]
fn should_never_exceed_the_budget_the_children_limit_or_the_lattice_depth() {
    let lattice = flat_lattice(4, 16, vec![0.0; 64]);
    let tree = best_first_draft_tree(&lattice, 0, &options(15, 4), None);
    assert_eq!(tree.token_ids.len(), 15);
    for depth in &tree.depths {
        assert!(*depth < 4);
    }
    let mut child_counts = std::collections::HashMap::new();
    for parent_index in &tree.parent_indices {
        *child_counts.entry(*parent_index).or_insert(0) += 1;
    }
    for child_count in child_counts.values() {
        assert!(*child_count <= 4);
    }
    for (node_index, parent_index) in tree.parent_indices.iter().enumerate() {
        if let Some(parent_index) = parent_index {
            assert!(*parent_index < node_index, "a parent precedes its children");
            assert_eq!(tree.depths[*parent_index] + 1, tree.depths[node_index]);
        }
    }
}

#[test]
fn should_let_the_parents_edge_reorder_a_slots_candidates() {
    let mut lattice = flat_lattice(2, 2, vec![5.0, 0.0, 1.0, 0.0]);
    // After first-slot candidate 0, the edge strongly favours candidate 1.
    lattice.edge_scores = vec![0.0, 30.0, 0.0, 0.0];
    let tree = best_first_draft_tree(&lattice, 0, &options(2, 1), None);
    assert_eq!(tree.token_ids, vec![0, 101]);
}

#[test]
fn should_lean_toward_the_token_the_keyed_noise_favours_at_its_position() {
    let lattice = flat_lattice(1, 3, vec![4.0, 0.0, 0.0]);
    let favoured = FavouredToken {
        position: 8,
        token_id: 2,
    };
    let tree = best_first_draft_tree(&lattice, 7, &options(1, 1), Some(&favoured));
    assert_eq!(tree.token_ids, vec![2]);
    let elsewhere = FavouredToken {
        position: 9,
        token_id: 2,
    };
    let tree = best_first_draft_tree(&lattice, 7, &options(1, 1), Some(&elsewhere));
    assert_eq!(tree.token_ids, vec![0]);
}

#[test]
fn should_return_an_empty_tree_for_no_budget_or_no_slots() {
    let lattice = flat_lattice(2, 2, vec![0.0; 4]);
    assert_eq!(
        best_first_draft_tree(&lattice, 0, &options(0, 2), None),
        DraftTree::default()
    );
    let empty_lattice = flat_lattice(0, 2, Vec::new());
    assert_eq!(
        best_first_draft_tree(&empty_lattice, 0, &options(5, 2), None),
        DraftTree::default()
    );
}

#[test]
fn should_draft_a_block_as_deep_as_the_node_budget_up_to_the_longest_block() {
    use super::draft_tree::tree_block_row_count;
    assert_eq!(tree_block_row_count(15, 16), 16);
    assert_eq!(tree_block_row_count(15, 8), 8);
    assert_eq!(tree_block_row_count(40, 16), 16);
    assert_eq!(tree_block_row_count(3, 8), 4);
    assert_eq!(tree_block_row_count(0, 8), 1);
}
