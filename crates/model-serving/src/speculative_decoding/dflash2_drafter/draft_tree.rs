//! The draft tree: a best-first search over the drafter's candidate lattice
//! for the branches most likely to match what the target emits.
//!
//! Ported from TensorFold's DFlash tree selector (release 1.2.0,
//! `zig/src/families/qwen27/dflash/selector.zig`, Apache-2.0, Copyright 2026
//! TensorFold contributors). One drafter pass scores `topk` candidates at each
//! of `depth` slots; the tree conditions each candidate on its parent through
//! the selector's low-rank edge term, so different branches rank the same
//! slot's candidates differently. Every node's score is a log-probability
//! over its slot's candidates, a path's score their sum, and the search keeps
//! the `node_budget` best paths, expanding each node's `children_per_node`
//! best children. Verifying a whole tree costs about what one path does (16
//! rows against 8 measured 62 against 59 ms of projections on the 27B), and a
//! branch that misses still leaves its siblings to match.

/// The candidate lattice of one drafted block, read back from the GPU once.
#[derive(Debug, Clone, PartialEq)]
pub struct DraftLattice {
    /// Slots in the block after the anchor.
    pub depth: usize,
    /// Candidates kept at each slot.
    pub candidate_count: usize,
    /// `[depth][candidate_count]` token ids.
    pub candidate_ids: Vec<u32>,
    /// `[depth][candidate_count]` the drafter head's logit for each candidate.
    pub unary_scores: Vec<f32>,
    /// `[candidate_count]` edge score of each first-slot candidate after the anchor.
    pub anchor_edge_scores: Vec<f32>,
    /// `[depth - 1][candidate_count][candidate_count]`: the edge score of slot
    /// `s + 1`'s candidate `j` after slot `s`'s candidate `i`, at `[s][i][j]`.
    pub edge_scores: Vec<f32>,
}

/// TensorFold's selector settings; `Default` gives the values its serial
/// loop uses on chips with matrix units (`loneSelection`: 16 children a node,
/// so an uncertain slot can spend the whole budget on siblings).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DraftTreeOptions {
    /// Nodes in the tree besides its root (the anchor): rows verified past it.
    pub node_budget: usize,
    pub children_per_node: usize,
    /// Divides every candidate's score before its slot's normalization:
    /// above 1 it flattens the slot, so deep paths are not starved.
    pub score_temperature: f64,
    /// Weight of the parent-conditioned edge term against the head's logit.
    pub edge_weight: f64,
    /// Weight of the keyed noise added to each score when sampling.
    pub noise_weight: f64,
    /// The request's sampling temperature, applied only with noise.
    pub sampling_temperature: f64,
}

impl Default for DraftTreeOptions {
    fn default() -> Self {
        Self {
            node_budget: 15,
            children_per_node: 16,
            score_temperature: 1.5,
            edge_weight: 0.6,
            noise_weight: 0.7,
            sampling_temperature: 1.0,
        }
    }
}

/// The keyed noise the target samples with, drawn for a token at an absolute
/// position; the tree adds the same draws so its branches lean toward what
/// the keyed sampler will pick.
pub trait KeyedNoise {
    fn draw(&self, absolute_position: u64, token_id: u32) -> f64;
}

/// Rows of the block TensorFold drafts for a tree, the anchor included: its
/// `tree_block`, twice the 8 rows the 27B's drafter trained on.
pub const TENSORFOLD_TREE_BLOCK_ROW_COUNT: usize = 16;

/// Rows a tree round's drafted block holds, the anchor included: as deep as
/// a tree of `node_budget` nodes can reach, up to `longest_block_row_count`.
///
/// The engine serves with the checkpoint's trained block. TensorFold's
/// 16-row block reproduces its rounds one for one on its own greedy code
/// completion (5.81 tokens kept a round against 5.62) and lifted raw code
/// completion from 65.7 to 72.0 tok/s on the 27B, but through the real
/// worker it kept 2.96 tokens a round instead of 3.73 on sampled prose after
/// a 10,011-token prompt (28.3 against 35.6 tok/s), a loss the model-level
/// journeys do not show and that is not yet explained.
#[must_use]
pub fn tree_block_row_count(node_budget: usize, longest_block_row_count: usize) -> usize {
    longest_block_row_count.min(node_budget + 1)
}

/// A tree of drafted tokens in the order they were chosen: every node comes
/// after its parent, so the rows can be verified in this order.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DraftTree {
    pub token_ids: Vec<u32>,
    /// Each node's parent index, or `None` when its parent is the anchor.
    pub parent_indices: Vec<Option<usize>>,
    /// Each node's slot: 0 for the anchor's children.
    pub depths: Vec<usize>,
}

/// A candidate waiting on the search frontier.
#[derive(Debug, Clone, Copy)]
struct FrontierNode {
    path_score: f64,
    parent_index: Option<usize>,
    candidate_index: usize,
    token_id: u32,
    depth: usize,
}

/// Where the next slot's candidates hang: under the anchor or a chosen node.
#[derive(Debug, Clone, Copy)]
struct ExpansionPoint {
    parent_index: Option<usize>,
    /// The parent's candidate index at its slot; None for the anchor.
    parent_candidate_index: Option<usize>,
    depth: usize,
    path_score: f64,
}

/// Searches `lattice` for the best `options.node_budget` nodes under the
/// anchor at `anchor_position`. With `keyed_noise`, slot `s` adds the draw at
/// `anchor_position + 1 + s`, the position the candidate would take.
pub fn best_first_draft_tree(
    lattice: &DraftLattice,
    anchor_position: u64,
    options: &DraftTreeOptions,
    keyed_noise: Option<&dyn KeyedNoise>,
) -> DraftTree {
    let scorer = SlotScorer {
        lattice,
        options,
        keyed_noise,
        anchor_position,
    };
    let mut frontier = Vec::new();
    let mut tree = DraftTree::default();
    if options.node_budget == 0 || lattice.depth == 0 {
        return tree;
    }
    scorer.expand(
        ExpansionPoint {
            parent_index: None,
            parent_candidate_index: None,
            depth: 0,
            path_score: 0.0,
        },
        &mut frontier,
    );
    while !frontier.is_empty() && tree.token_ids.len() < options.node_budget {
        let chosen = frontier.swap_remove(best_frontier_index(&frontier));
        let chosen_index = tree.token_ids.len();
        tree.token_ids.push(chosen.token_id);
        tree.parent_indices.push(chosen.parent_index);
        tree.depths.push(chosen.depth);
        if chosen.depth + 1 < lattice.depth {
            scorer.expand(
                ExpansionPoint {
                    parent_index: Some(chosen_index),
                    parent_candidate_index: Some(chosen.candidate_index),
                    depth: chosen.depth + 1,
                    path_score: chosen.path_score,
                },
                &mut frontier,
            );
        }
    }
    tree
}

/// The best frontier node: highest path score, ties to the earlier parent
/// (the anchor first), then the lower token id, as TensorFold orders them.
fn best_frontier_index(frontier: &[FrontierNode]) -> usize {
    let mut best_index = 0;
    for (candidate_index, candidate) in frontier.iter().enumerate().skip(1) {
        if ranks_before(candidate, &frontier[best_index]) {
            best_index = candidate_index;
        }
    }
    best_index
}

fn ranks_before(left: &FrontierNode, right: &FrontierNode) -> bool {
    if left.path_score != right.path_score {
        return left.path_score > right.path_score;
    }
    let (left_parent, right_parent) = (parent_order(left), parent_order(right));
    if left_parent != right_parent {
        return left_parent < right_parent;
    }
    left.token_id < right.token_id
}

/// The anchor sorts before every chosen node.
fn parent_order(node: &FrontierNode) -> i64 {
    match node.parent_index {
        Some(parent_index) => parent_index as i64,
        None => -1,
    }
}

/// The indices of the `count` highest scores, best first, ties to the lower
/// index. A slot holds 16 candidates, so a selection pass per pick is cheap.
fn best_candidates(slot_scores: &[f64], count: usize) -> Vec<usize> {
    let mut is_taken = vec![false; slot_scores.len()];
    let mut picked = Vec::with_capacity(count.min(slot_scores.len()));
    while picked.len() < count.min(slot_scores.len()) {
        let mut best: Option<usize> = None;
        for (candidate_index, score) in slot_scores.iter().enumerate() {
            let beats_best = match best {
                None => true,
                Some(best_index) => *score > slot_scores[best_index],
            };
            if !is_taken[candidate_index] && beats_best {
                best = Some(candidate_index);
            }
        }
        let Some(best_index) = best else { break };
        is_taken[best_index] = true;
        picked.push(best_index);
    }
    picked
}

struct SlotScorer<'search> {
    lattice: &'search DraftLattice,
    options: &'search DraftTreeOptions,
    keyed_noise: Option<&'search dyn KeyedNoise>,
    anchor_position: u64,
}

impl SlotScorer<'_> {
    /// Adds the best `children_per_node` candidates of the point's slot to
    /// the frontier, each scored as a log-probability over its slot.
    fn expand(&self, point: ExpansionPoint, frontier: &mut Vec<FrontierNode>) {
        let slot_scores = self.normalized_slot_scores(point);
        let row_start = point.depth * self.lattice.candidate_count;
        for candidate_index in best_candidates(&slot_scores, self.options.children_per_node) {
            frontier.push(FrontierNode {
                path_score: point.path_score + slot_scores[candidate_index],
                parent_index: point.parent_index,
                candidate_index,
                token_id: self.lattice.candidate_ids[row_start + candidate_index],
                depth: point.depth,
            });
        }
    }

    /// `(unary + edge_weight * edge) / temperature + noise_weight * noise`,
    /// over `score_temperature`, then log-softmax over the slot.
    fn normalized_slot_scores(&self, point: ExpansionPoint) -> Vec<f64> {
        let candidate_count = self.lattice.candidate_count;
        let temperature = match self.keyed_noise {
            Some(_) => self.options.sampling_temperature.max(1e-6),
            None => 1.0,
        };
        let row_start = point.depth * candidate_count;
        let mut scores = Vec::with_capacity(candidate_count);
        for candidate_index in 0..candidate_count {
            let unary = f64::from(self.lattice.unary_scores[row_start + candidate_index]);
            let edge = self.edge_score(point, candidate_index);
            let mut score = unary / temperature + self.options.edge_weight * edge / temperature;
            if let Some(keyed_noise) = self.keyed_noise {
                let candidate_position = self.anchor_position + 1 + point.depth as u64;
                let token_id = self.lattice.candidate_ids[row_start + candidate_index];
                score += self.options.noise_weight * keyed_noise.draw(candidate_position, token_id);
            }
            scores.push(score / self.options.score_temperature);
        }
        log_softmax_in_place(&mut scores);
        scores
    }

    fn edge_score(&self, point: ExpansionPoint, candidate_index: usize) -> f64 {
        let candidate_count = self.lattice.candidate_count;
        let edge = match point.parent_candidate_index {
            None => self.lattice.anchor_edge_scores[candidate_index],
            Some(parent_candidate_index) => {
                let table_start = (point.depth - 1) * candidate_count * candidate_count;
                self.lattice.edge_scores
                    [table_start + parent_candidate_index * candidate_count + candidate_index]
            }
        };
        f64::from(edge)
    }
}

fn log_softmax_in_place(scores: &mut [f64]) {
    let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut exponential_sum = 0.0;
    for score in scores.iter_mut() {
        *score -= maximum;
        exponential_sum += score.exp();
    }
    let normalizer = exponential_sum.ln();
    for score in scores.iter_mut() {
        *score -= normalizer;
    }
}
