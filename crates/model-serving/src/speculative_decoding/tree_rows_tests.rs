use super::dflash2_drafter::DraftTree;
use super::tree_rows::{KeptChain, TreeRows};

/// Anchor 7 with two children (10, 11); 10 has child 20, 20 has child 30.
fn branching_tree() -> TreeRows {
    TreeRows::from_draft_tree(
        7,
        &DraftTree {
            token_ids: vec![10, 11, 20, 30],
            parent_indices: vec![None, None, Some(0), Some(2)],
            depths: vec![0, 0, 1, 2],
        },
    )
}

#[test]
fn should_put_the_anchor_first_and_each_node_one_row_below_its_parent() {
    let rows = branching_tree();
    assert_eq!(rows.token_ids, vec![7, 10, 11, 20, 30]);
    assert_eq!(
        rows.parent_rows,
        vec![None, Some(0), Some(0), Some(1), Some(3)]
    );
    assert_eq!(rows.depths, vec![0, 1, 1, 2, 3]);
}

#[test]
fn should_give_siblings_the_same_position() {
    assert_eq!(
        branching_tree().positions(100),
        vec![100, 101, 101, 102, 103]
    );
}

#[test]
fn should_let_a_row_see_only_itself_and_its_ancestors() {
    let rows = branching_tree();
    let visibility = rows.ancestor_visibility();
    let visible = |row: usize| -> Vec<usize> {
        (0..5)
            .filter(|column| visibility[row * 5 + column])
            .collect()
    };
    assert_eq!(visible(0), vec![0]);
    assert_eq!(visible(2), vec![0, 2]);
    assert_eq!(visible(4), vec![0, 1, 3, 4]);
}

#[test]
fn should_read_each_convolution_window_down_its_own_chain_then_the_state() {
    // Window 4: state rows 0..3 precede tree rows, which start at index 3.
    let sources = branching_tree().convolution_window_sources(4);
    assert_eq!(&sources[0..4], &[0, 1, 2, 3]); // anchor: whole state, itself
    assert_eq!(&sources[8..12], &[1, 2, 3, 5]); // row 2: sibling 1 is not read
    assert_eq!(&sources[16..20], &[3, 4, 6, 7]); // row 4: its full chain
}

#[test]
fn should_keep_the_deepest_chain_the_targets_picks_follow() {
    let rows = branching_tree();
    // After the anchor the target picks 10, then 20, then 99 (no such child).
    let picks = [10, 0, 0, 99, 0];
    let kept = rows.kept_chain(&[picks[0], 20, 0, picks[3], 0]);
    assert_eq!(
        kept,
        KeptChain {
            rows: vec![0, 1, 3],
            next_token_id: 99
        }
    );
}

#[test]
fn should_keep_only_the_anchor_when_the_first_pick_matches_no_child() {
    let kept = branching_tree().kept_chain(&[42, 0, 0, 0, 0]);
    assert_eq!(
        kept,
        KeptChain {
            rows: vec![0],
            next_token_id: 42
        }
    );
}

#[test]
fn should_follow_a_whole_branch_and_return_the_pick_after_its_leaf() {
    let kept = branching_tree().kept_chain(&[10, 20, 0, 30, 55]);
    assert_eq!(kept.rows, vec![0, 1, 3, 4]);
    assert_eq!(kept.next_token_id, 55);
}

#[test]
fn should_treat_an_empty_tree_as_the_anchor_alone() {
    let rows = TreeRows::from_draft_tree(5, &DraftTree::default());
    assert_eq!(rows.row_count(), 1);
    assert_eq!(rows.convolution_window_sources(4), vec![0, 1, 2, 3]);
    assert_eq!(rows.kept_chain(&[8]).rows, vec![0]);
}

#[test]
fn should_share_agreeing_rows_with_a_copy_and_chain_the_rest_below_them() {
    let mut rows = branching_tree();
    // 10 and 20 are already rows 1 and 3; 40 and 50 hang below row 3.
    assert_eq!(rows.add_copied_chain(&[10, 20, 40, 50]), 2);
    assert_eq!(rows.token_ids[5..], [40, 50]);
    assert_eq!(rows.parent_rows[5..], [Some(3), Some(5)]);
    assert_eq!(rows.depths[5..], [3, 4]);
}

#[test]
fn should_add_a_copy_that_disagrees_at_once_as_a_new_branch_under_the_anchor() {
    let mut rows = branching_tree();
    assert_eq!(rows.add_copied_chain(&[99, 98]), 2);
    assert_eq!(rows.parent_rows[5..], [Some(0), Some(5)]);
}

#[test]
fn should_add_no_rows_for_a_copy_the_tree_already_holds_or_an_empty_one() {
    let mut rows = branching_tree();
    assert_eq!(rows.add_copied_chain(&[10, 20, 30]), 0);
    assert_eq!(rows.add_copied_chain(&[]), 0);
    assert_eq!(rows.row_count(), 5);
}
