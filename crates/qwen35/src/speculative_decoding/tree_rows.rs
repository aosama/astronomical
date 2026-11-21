//! A draft tree as the rows one target pass verifies: row 0 is the anchor
//! (the last emitted token, not yet fed), and every tree node follows its
//! parent. This decides what each row may see and where it sits, and, once
//! the target has picked a token after every row, which chain is kept.
//!
//! A row sits one position past its parent, so siblings share a position.
//! Attention lets a row see the context and its own ancestors only, and a
//! causal convolution over the last `window` inputs reads the row's own
//! ancestor chain, reaching back into the convolution state for the inputs
//! before the anchor. With those, every row computes exactly what it would
//! have computed had its chain been fed alone.

use super::dflash_drafter::DraftTree;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TreeRows {
    /// The token each row feeds: the anchor, then the tree's nodes in order.
    pub token_ids: Vec<u32>,
    /// Each row's parent row; None for the anchor.
    pub parent_rows: Vec<Option<usize>>,
    /// Rows from the anchor: 0 for the anchor, 1 for its children.
    pub depths: Vec<usize>,
}

/// What a round keeps once the target has picked after every row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeptChain {
    /// The kept rows from the anchor down, in order: their inputs are now
    /// part of the sequence.
    pub rows: Vec<usize>,
    /// The target's pick after the last kept row: the round's bonus token.
    pub next_token_id: u32,
}

impl TreeRows {
    pub fn from_draft_tree(anchor_token_id: u32, tree: &DraftTree) -> Self {
        let mut rows = Self {
            token_ids: vec![anchor_token_id],
            parent_rows: vec![None],
            depths: vec![0],
        };
        for (node_index, token_id) in tree.token_ids.iter().enumerate() {
            let parent_row = match tree.parent_indices[node_index] {
                Some(parent_node) => parent_node + 1,
                None => 0,
            };
            rows.token_ids.push(*token_id);
            rows.parent_rows.push(Some(parent_row));
            rows.depths.push(rows.depths[parent_row] + 1);
        }
        rows
    }

    /// Hangs `copied` (tokens proposed to follow the anchor, in order) under
    /// the anchor: it shares the tree's rows while they agree, and the rest
    /// becomes a chain below the last shared row. Returns the rows added.
    pub fn add_copied_chain(&mut self, copied: &[u32]) -> usize {
        let first_new_row = self.row_count();
        let mut row = 0;
        for token_id in copied {
            let mut matching_child = None;
            for (child_row, parent_row) in self.parent_rows.iter().enumerate() {
                if *parent_row == Some(row) && self.token_ids[child_row] == *token_id {
                    matching_child = Some(child_row);
                    break;
                }
            }
            row = match matching_child {
                Some(child_row) => child_row,
                None => {
                    self.token_ids.push(*token_id);
                    self.parent_rows.push(Some(row));
                    self.depths.push(self.depths[row] + 1);
                    self.row_count() - 1
                }
            };
        }
        self.row_count() - first_new_row
    }

    pub fn row_count(&self) -> usize {
        self.token_ids.len()
    }

    /// Each row's absolute position: the anchor's plus its depth.
    pub fn positions(&self, anchor_position: i32) -> Vec<i32> {
        let mut positions = Vec::with_capacity(self.row_count());
        for depth in &self.depths {
            positions.push(anchor_position + *depth as i32);
        }
        positions
    }

    /// `[rows][rows]`, row-major: whether row `i` may attend to row `j`,
    /// which holds when `j` is `i` or one of its ancestors.
    pub fn ancestor_visibility(&self) -> Vec<bool> {
        let row_count = self.row_count();
        let mut visibility = vec![false; row_count * row_count];
        for row in 0..row_count {
            let mut ancestor = Some(row);
            while let Some(visible_row) = ancestor {
                visibility[row * row_count + visible_row] = true;
                ancestor = self.parent_rows[visible_row];
            }
        }
        visibility
    }

    /// `[rows][window]`: the inputs each row's causal convolution reads,
    /// oldest first, as indices into `window - 1` state rows followed by the
    /// tree's rows. A row's window is its ancestor chain, then the state.
    pub fn convolution_window_sources(&self, window: usize) -> Vec<i32> {
        let state_row_count = window - 1;
        let mut sources = Vec::with_capacity(self.row_count() * window);
        for row in 0..self.row_count() {
            let mut reversed_window = Vec::with_capacity(window);
            let mut chain_row = Some(row);
            let mut state_row = state_row_count;
            while reversed_window.len() < window {
                match chain_row {
                    Some(current_row) => {
                        reversed_window.push((state_row_count + current_row) as i32);
                        chain_row = self.parent_rows[current_row];
                    }
                    None => {
                        state_row -= 1;
                        reversed_window.push(state_row as i32);
                    }
                }
            }
            reversed_window.reverse();
            sources.extend(reversed_window);
        }
        sources
    }

    /// The chain the target's `picks` (one per row: its token after that
    /// row) run down from the anchor, and the pick after its last row.
    pub fn kept_chain(&self, picks: &[u32]) -> KeptChain {
        let mut rows = vec![0];
        loop {
            let last_row = *rows.last().expect("the chain holds the anchor");
            let pick = picks[last_row];
            let mut matching_child = None;
            for (row, parent_row) in self.parent_rows.iter().enumerate() {
                if *parent_row == Some(last_row) && self.token_ids[row] == pick {
                    matching_child = Some(row);
                    break;
                }
            }
            match matching_child {
                Some(child_row) => rows.push(child_row),
                None => {
                    return KeptChain {
                        rows,
                        next_token_id: pick,
                    };
                }
            }
        }
    }
}
