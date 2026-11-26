//! Gate 4 end to end on a real target and its DFlash2 drafter (the 27B pair
//! unless the test run configures a target with its own `drafter` folder):
//! every round drafts a tree, verifies all of it in one target pass, picks
//! with the keyed sampler (or greedily), and commits the kept chain. Prints
//! kept per round and the decode speed (prefill excluded) for TensorFold's
//! code-completion cell and the story (`story_cell`). On the 27B pair
//! TensorFold's own engine measured 78.7 tok/s on that code cell, on the Mac
//! these journeys were first run on; compare against a TensorFold run on the
//! Mac at hand.
//!
//! Each stage still waits for the GPU (the lattice and the picks are read
//! back), so this is the speed a straightforward tree loop reaches, not a
//! pipelined serving loop's.

use model_serving::DraftTreeTarget;
use std::time::Instant;

use mlx_bridge::MlxArray;
use model_serving::RequestDecoderStateStack;
use model_serving::speculative_decoding::dflash_drafter::DraftTreeOptions;
use model_serving::speculative_decoding::tree_rows::TreeRows;
use serial_test::serial;

use model_serving::speculative_decoding::context_copy::{ContextCopier, ContextCopySettings};

use super::dflash2_journey_support::{TargetLogitRows, forward_with_taps, story_cell};
use super::dflash2_kept_per_round::{
    Cell, ENGINE_CHILDREN_PER_NODE, ENGINE_NODE_BUDGET, JourneyPair, OUTPUT_TOKENS, RoundState,
    completion_token_ids, load_pair, prompt_token_ids,
};
use super::dflash2_tree_oracle::{
    Drafting, FIRST_KEYED_SEED, ORACLE_CELLS, draft_tree, prefill, target_picks,
};

/// Seeds pooled for a sampled cell, as TensorFold's three repetitions.
const SAMPLED_SEED_COUNT: u64 = 3;
/// TensorFold's copied-row budget: it starts at 16 rows past the tree,
/// doubles while copies land whole, halves when most miss, and stops at 48.
const FIRST_COPY_ROWS: usize = 16;
const MAXIMUM_COPY_ROWS: usize = 48;

#[test]
#[ignore = "loads a real target and its DFlash2 drafter (the 27B pair unless configured); run through cargo measure-dflash2-tree-rounds"]
#[serial]
fn should_decode_code_cells_with_draft_trees_verified_in_one_target_pass() {
    decode_cells_with_trees(&ORACLE_CELLS[..3], SAMPLED_SEED_COUNT);
}

/// A story continued after a long prompt, sampled: the cell a long chat or a
/// document question lands in, where few drafted tokens are kept a round
/// and every verified row attends to the whole prompt.
#[test]
#[ignore = "loads a real target and its DFlash2 drafter (the 27B pair unless configured); run through cargo measure-dflash2-tree-rounds"]
#[serial]
fn should_decode_a_story_after_a_long_prompt_with_draft_trees() {
    // One seed: every decode reads the story prompt again.
    decode_cells_with_trees(&[story_cell("story after a long prompt, 1.0", 1_000)], 1);
}

/// Node budgets for the tree-size ladder: 2- to 8-row rounds, where the
/// first ladder on the 35B-A3B target (3 to 15 nodes) peaked at its
/// smallest size.
const LADDER_NODE_BUDGETS: [usize; 6] = [1, 2, 3, 4, 5, 7];
/// Seeds pooled at each budget of the ladder.
const LADDER_SEED_COUNT: u64 = 2;

/// The sampled story at a ladder of tree sizes. On a mixture-of-experts
/// target every verified row pays its own expert reads, so a round's cost
/// grows with its rows while the tokens it keeps level off; this finds the
/// size where the two meet. One process, smallest first.
#[test]
#[ignore = "loads a real target and its DFlash2 drafter; run through cargo measure-dflash2-tree-rounds"]
#[serial]
fn should_decode_the_story_at_a_ladder_of_tree_sizes() {
    report_plain_decode(&story_cell("story 1.0", 1_000), LADDER_SEED_COUNT);
    let mut cells = Vec::with_capacity(LADDER_NODE_BUDGETS.len());
    for node_budget in LADDER_NODE_BUDGETS {
        let mut cell = story_cell("story 1.0", 1_000);
        cell.label = Box::leak(format!("story 1.0, {node_budget}-node tree").into_boxed_str());
        cell.node_budget = node_budget;
        cells.push(cell);
    }
    decode_cells_with_trees(&cells, LADDER_SEED_COUNT);
}

/// The serving journey's own cell in this harness: about 10,000 prompt
/// tokens, a story at temperature 1.0, plain against the small trees a
/// mixture-of-experts target drafts with. Read against the journey: if this
/// keeps the short prompt's gain, the serving loop loses it; if it does not,
/// the long prompt does.
#[test]
#[ignore = "loads a real target and its DFlash2 drafter; run through cargo measure-dflash2-tree-rounds"]
#[serial]
fn should_decode_a_story_after_a_journey_sized_prompt_plain_and_with_small_trees() {
    let mut cell = story_cell("story 1.0", 1_000);
    cell.label = "story after a journey-sized prompt, 1.0";
    report_plain_decode(&cell, 1);
    let mut cells = Vec::new();
    for node_budget in [3, 7] {
        let mut sized_cell = cell;
        sized_cell.label = Box::leak(
            format!("story after a journey-sized prompt, 1.0, {node_budget}-node tree")
                .into_boxed_str(),
        );
        sized_cell.node_budget = node_budget;
        cells.push(sized_cell);
    }
    decode_cells_with_trees(&cells, 1);
}

/// Seeds pooled for the kernel comparison: at temperature 1.0 any change in
/// arithmetic sends the sampled text down another path, so one seed cannot
/// tell a kernel's effect on kept tokens from the text's.
const KERNEL_COMPARISON_SEED_COUNT: u64 = 6;

/// The 3-node tree after a journey-sized prompt with the row-exact tree
/// attention kernel and with MLX's masked attention, seeds pooled, in one
/// process.
#[test]
#[ignore = "loads a real target and its DFlash2 drafter; run through cargo measure-dflash2-tree-rounds"]
#[serial]
fn should_compare_the_tree_attention_kernel_with_mlx_after_a_journey_sized_prompt() {
    let mut cell = story_cell("story 1.0", 1_000);
    cell.node_budget = 3;
    cell.label = "story after a journey-sized prompt, 1.0, 3-node tree";
    compare_tree_attention_paths(
        &cell,
        JOURNEY_NOISE_CHECK_SEED_COUNT,
        &[
            TreeAttentionPath::RowExactKernel,
            TreeAttentionPath::MlxBf16,
        ],
    );
}

/// The same comparison on the story prompt at every seed.
#[test]
#[ignore = "loads a real target and its DFlash2 drafter; run through cargo measure-dflash2-tree-rounds"]
#[serial]
fn should_compare_the_tree_attention_kernel_with_mlx_after_a_short_prompt() {
    let mut cell = story_cell("story 1.0", 1_000);
    cell.node_budget = 3;
    cell.label = "story 1.0, 3-node tree";
    compare_tree_attention_per_seed(&cell, KERNEL_COMPARISON_SEED_COUNT);
}

/// How a tree's full attention runs in a comparison.
#[derive(Clone, Copy)]
enum TreeAttentionPath {
    RowExactKernel,
    MlxBf16,
    MlxFloat32,
}

impl TreeAttentionPath {
    const ALL: [Self; 3] = [Self::RowExactKernel, Self::MlxBf16, Self::MlxFloat32];

    fn label(self) -> &'static str {
        match self {
            Self::RowExactKernel => "row-exact kernel",
            Self::MlxBf16 => "MLX attention, bf16 (serving)",
            Self::MlxFloat32 => "MLX attention, float32",
        }
    }

    fn configure(self, pair: &mut JourneyPair) {
        pair.target
            .enable_draft_tree_verification()
            .expect("the tree kernels declare");
        match self {
            Self::RowExactKernel => pair
                .target
                .enable_row_exact_tree_attention()
                .expect("the tree attention kernel declares"),
            Self::MlxBf16 | Self::MlxFloat32 => pair.target.set_row_exact_tree_attention_aside(),
        }
        pair.target
            .set_tree_attention_in_float32_for_measurement(matches!(self, Self::MlxFloat32));
    }
}

/// Seeds for the short-prompt noise check: enough that a difference of a
/// tenth of a kept token a round stands out from the seeds' own spread.
const NOISE_CHECK_SEED_COUNT: u64 = 24;
/// The same after a journey-sized prompt, where each seed prefills 10,000
/// tokens: half the seeds keep the run near four minutes.
const JOURNEY_NOISE_CHECK_SEED_COUNT: u64 = 12;

/// The kernel against serving's attention over many seeds of the story, each seed printed, to tell the kernel's effect on kept
/// tokens from the spread between sampled texts.
#[test]
#[ignore = "loads a real target and its DFlash2 drafter; run through cargo measure-dflash2-tree-rounds"]
#[serial]
fn should_tell_the_kernels_effect_on_kept_tokens_from_the_spread_between_seeds() {
    let mut cell = story_cell("story 1.0", 1_000);
    cell.node_budget = 3;
    cell.label = "story 1.0, 3-node tree";
    compare_tree_attention_paths(
        &cell,
        NOISE_CHECK_SEED_COUNT,
        &[
            TreeAttentionPath::RowExactKernel,
            TreeAttentionPath::MlxBf16,
        ],
    );
}

/// Decodes `cell` once a seed through every tree attention path, alternating
/// in one process so heat and drift hit all alike, and prints each path
/// pooled over the seeds.
fn compare_tree_attention_per_seed(cell: &Cell, seed_count: u64) {
    compare_tree_attention_paths(cell, seed_count, &TreeAttentionPath::ALL);
}

/// Like `compare_tree_attention_per_seed` for chosen paths, also printing
/// each seed's kept tokens a round, whose spread is the noise a difference
/// between the pooled figures has to clear.
fn compare_tree_attention_paths(cell: &Cell, seed_count: u64, paths: &[TreeAttentionPath]) {
    let mut pair = load_pair();
    let mut pooled: Vec<PooledTreeDecode> =
        paths.iter().map(|_| PooledTreeDecode::default()).collect();
    for seed_offset in 0..seed_count {
        let seeded_cell = Cell {
            keyed_seed: FIRST_KEYED_SEED + seed_offset,
            ..*cell
        };
        let mut seed_line = format!("[dflash2-tree-rounds] seed {seed_offset:>2}:");
        for (pooled_decode, path) in pooled.iter_mut().zip(paths) {
            path.configure(&mut pair);
            let decode = decode_with_trees(&pair, &seeded_cell, false);
            seed_line.push_str(&format!(
                " {} {:.2}",
                path.label(),
                (decode.emitted_token_ids.len() - 1) as f64 / decode.rounds as f64
            ));
            pooled_decode.add(&decode);
        }
        eprintln!("{seed_line}");
    }
    for (pooled_decode, path) in pooled.iter().zip(paths.iter().copied()) {
        print_pooled(
            &format!("{}, {}", cell.label, path.label()),
            seed_count,
            pooled_decode,
        );
    }
}

/// One pooled decode's kept tokens a round, speed and round time.
fn print_pooled(label: &str, seed_count: u64, pooled_decode: &PooledTreeDecode) {
    eprintln!(
        "[dflash2-tree-rounds] {label} ({seed_count} seeds): {:.2} kept per round, {:.1} tok/s decode ({} tokens in {} rounds, {:.1} ms a round)",
        pooled_decode.emitted as f64 / pooled_decode.rounds as f64,
        pooled_decode.emitted as f64 / pooled_decode.decode_seconds,
        pooled_decode.emitted,
        pooled_decode.rounds,
        pooled_decode.decode_seconds * 1000.0 / pooled_decode.rounds as f64,
    );
}

/// Seeds for the depth comparison, interleaved per seed.
const DEPTH_COMPARISON_SEED_COUNT: u64 = 16;

/// Looking four tokens ahead against today's three, on the story at
/// temperature 1.0: the 3-node tree serving uses, a 4-node tree (the same
/// search, reaching depth 4 where the drafter is confident) and a 4-deep
/// chain (one candidate a node, always depth 4), each seed through all three,
/// with plain one-token steps as the baseline.
#[test]
#[ignore = "loads a real target and its DFlash2 drafter; run through cargo measure-dflash2-tree-rounds"]
#[serial]
fn should_compare_drafting_three_and_four_tokens_ahead() {
    report_plain_decode(&story_cell("story 1.0", 1_000), DEPTH_COMPARISON_SEED_COUNT);
    let mut cells = Vec::new();
    for (label, node_budget, children_per_node) in [
        ("3-node tree (serving)", 3, ENGINE_CHILDREN_PER_NODE),
        ("4-node tree", 4, ENGINE_CHILDREN_PER_NODE),
        ("4-deep chain", 4, 1),
    ] {
        let mut cell = story_cell("story 1.0", 1_000);
        cell.label = label;
        cell.node_budget = node_budget;
        cell.children_per_node = children_per_node;
        cells.push(cell);
    }
    compare_cells_per_seed(&cells, DEPTH_COMPARISON_SEED_COUNT);
}

/// The 3-node tree serving uses against a 3-node chain: the same 4 rows a
/// round, the chain always reaching 3 drafted tokens deep (4 counting the
/// round's own token), the tree spending nodes on alternatives where the
/// drafter is unsure.
#[test]
#[ignore = "loads a real target and its DFlash2 drafter; run through cargo measure-dflash2-tree-rounds"]
#[serial]
fn should_compare_a_three_node_tree_with_a_three_node_chain() {
    let mut cells = Vec::new();
    for (label, children_per_node) in [
        ("3-node tree (serving)", ENGINE_CHILDREN_PER_NODE),
        ("3-node chain", 1),
    ] {
        let mut cell = story_cell("story 1.0", 1_000);
        cell.label = label;
        cell.node_budget = 3;
        cell.children_per_node = children_per_node;
        cells.push(cell);
    }
    compare_cells_per_seed(&cells, DEPTH_COMPARISON_SEED_COUNT);
}

/// Decodes every cell once a seed, interleaved so heat and drift hit all
/// alike, printing each seed's kept tokens a round and each cell pooled.
fn compare_cells_per_seed(cells: &[Cell], seed_count: u64) {
    // The cells' engine figures are copied into constants because the cells
    // are built at compile time; they must still be the engine's own.
    let engine_options = DraftTreeOptions::default();
    assert_eq!(ENGINE_NODE_BUDGET, engine_options.node_budget);
    assert_eq!(ENGINE_CHILDREN_PER_NODE, engine_options.children_per_node);
    let mut pair = load_pair();
    pair.target
        .enable_draft_tree_verification()
        .expect("the tree kernels declare");
    let mut pooled: Vec<PooledTreeDecode> =
        cells.iter().map(|_| PooledTreeDecode::default()).collect();
    for seed_offset in 0..seed_count {
        let mut seed_line = format!("[dflash2-tree-rounds] seed {seed_offset:>2}:");
        for (pooled_decode, cell) in pooled.iter_mut().zip(cells) {
            let seeded_cell = Cell {
                keyed_seed: FIRST_KEYED_SEED + seed_offset,
                ..*cell
            };
            let decode = decode_with_trees(&pair, &seeded_cell, false);
            seed_line.push_str(&format!(
                " | {} {:.2}",
                cell.label,
                (decode.emitted_token_ids.len() - 1) as f64 / decode.rounds as f64
            ));
            pooled_decode.add(&decode);
        }
        eprintln!("{seed_line}");
    }
    for (pooled_decode, cell) in pooled.iter().zip(cells) {
        print_pooled(cell.label, seed_count, pooled_decode);
    }
}

/// Plain decoding in the same harness, the ladder's baseline: one token a
/// step through the target's verification forward with no taps and no
/// drafter, picked as the tree rounds pick, over the same output length.
pub(super) fn report_plain_decode(cell: &Cell, seed_count: u64) {
    let pair = load_pair();
    let (mut emitted_count, mut decode_seconds) = (0, 0.0);
    for seed_offset in 0..seed_count {
        let seeded_cell = Cell {
            keyed_seed: FIRST_KEYED_SEED + seed_offset,
            ..*cell
        };
        let (round, mut decoder_state, first_token_id) = prefill(&pair, &seeded_cell);
        let mut last_token_id = first_token_id;
        let mut fed_token_count = round.fed_token_count;
        let decode_started_at = Instant::now();
        for _ in 0..OUTPUT_TOKENS {
            let step = forward_with_taps(
                &pair.target,
                &[last_token_id],
                fed_token_count,
                &mut decoder_state,
                &[],
                TargetLogitRows::EveryRow,
            );
            let logits = step.logits;
            fed_token_count += 1;
            last_token_id =
                target_picks(&pair, &seeded_cell, &logits, &[u64::from(fed_token_count)])[0];
        }
        decode_seconds += decode_started_at.elapsed().as_secs_f64();
        emitted_count += OUTPUT_TOKENS;
    }
    eprintln!(
        "[dflash2-tree-rounds] {} ({seed_count} seeds), plain one-token steps: {:.1} tok/s decode ({emitted_count} tokens, {:.1} ms a token)",
        cell.label,
        emitted_count as f64 / decode_seconds,
        decode_seconds * 1000.0 / emitted_count as f64,
    );
}

/// Decodes every cell with and without context copies; a sampled cell
/// pools up to `sampled_seed_limit` seeds.
pub(super) fn decode_cells_with_trees(cells: &[Cell], sampled_seed_limit: u64) {
    let mut pair = load_pair();
    pair.target
        .enable_draft_tree_verification()
        .expect("the tree kernel declares");
    for (cell_index, cell) in cells.iter().enumerate() {
        let seed_count = if cell.temperature_thousandths == 0 {
            1
        } else {
            SAMPLED_SEED_COUNT.min(sampled_seed_limit)
        };
        // Alternating the two in one process keeps drift out of the comparison.
        let mut pooled = [PooledTreeDecode::default(), PooledTreeDecode::default()];
        for seed_offset in 0..seed_count {
            let seeded_cell = Cell {
                keyed_seed: FIRST_KEYED_SEED + seed_offset,
                ..*cell
            };
            for (pooled_decode, copies_context) in pooled.iter_mut().zip([false, true]) {
                pooled_decode.add(&decode_with_trees(&pair, &seeded_cell, copies_context));
            }
        }
        for (pooled_decode, label) in pooled.iter().zip(["tree", "tree + context copies"]) {
            eprintln!(
                "[dflash2-tree-rounds {}/{}] {} ({seed_count} seeds), {label}: {:.2} kept per \
                 round, {:.1} tok/s decode ({} tokens in {} rounds, {:.1} ms a round)",
                cell_index + 1,
                cells.len(),
                cell.label,
                pooled_decode.emitted as f64 / pooled_decode.rounds as f64,
                pooled_decode.emitted as f64 / pooled_decode.decode_seconds,
                pooled_decode.emitted,
                pooled_decode.rounds,
                pooled_decode.decode_seconds * 1000.0 / pooled_decode.rounds as f64,
            );
        }
        print_segments(pooled[0].rounds + pooled[1].rounds);
    }
}

/// Tree decodes pooled over seeds.
#[derive(Default)]
struct PooledTreeDecode {
    rounds: usize,
    emitted: usize,
    decode_seconds: f64,
}

impl PooledTreeDecode {
    fn add(&mut self, tree: &TreeDecode) {
        self.rounds += tree.rounds;
        self.emitted += tree.emitted_token_ids.len() - 1;
        self.decode_seconds += tree.decode_seconds;
    }
}

struct TreeDecode {
    rounds: usize,
    /// The prefill's pick, then every token the rounds kept.
    emitted_token_ids: Vec<u32>,
    decode_seconds: f64,
}

/// The request's text so far and its copier, when the round adds copies.
struct CopyContext {
    history: Vec<u32>,
    copier: ContextCopier,
    copy_row_budget: usize,
}

fn decode_with_trees(pair: &JourneyPair, cell: &Cell, copies_context: bool) -> TreeDecode {
    let (mut round, mut decoder_state, first_token_id) = prefill(pair, cell);
    let mut copy_context = if copies_context {
        let mut history = if cell.is_raw_completion {
            completion_token_ids(pair, cell.prompt)
        } else {
            prompt_token_ids(pair, cell.prompt)
        };
        history.push(first_token_id);
        Some(CopyContext {
            history,
            copier: ContextCopier::new(ContextCopySettings::default()),
            copy_row_budget: FIRST_COPY_ROWS,
        })
    } else {
        None
    };
    let decode_started_at = Instant::now();
    let mut emitted_token_ids = vec![first_token_id];
    let mut rounds = 0;
    while emitted_token_ids.len() <= OUTPUT_TOKENS {
        let emitted = tree_round(
            pair,
            cell,
            &mut round,
            &mut decoder_state,
            copy_context.as_mut(),
        );
        if let Some(copy_context) = copy_context.as_mut() {
            copy_context.history.extend_from_slice(&emitted);
        }
        emitted_token_ids.extend(emitted);
        rounds += 1;
    }
    TreeDecode {
        rounds,
        emitted_token_ids,
        decode_seconds: decode_started_at.elapsed().as_secs_f64(),
    }
}

/// Drafts a tree, hangs a context copy below it when given a copier,
/// verifies every row, keeps the chain the target's picks follow and
/// commits it. Returns the tokens the round emitted.
fn tree_round(
    pair: &JourneyPair,
    cell: &Cell,
    round: &mut RoundState,
    decoder_state: &mut RequestDecoderStateStack,
    mut copy_context: Option<&mut CopyContext>,
) -> Vec<u32> {
    let segment_started_at = Instant::now();
    let tree = draft_tree(pair, cell, round, Drafting::Tree);
    let segment_started_at = record_segment(0, segment_started_at);
    let mut rows = TreeRows::from_draft_tree(round.last_token_id, &tree);
    let first_copied_row = rows.row_count();
    let mut copied_row_count = 0;
    if let Some(copy_context) = copy_context.as_deref_mut() {
        let copied = copy_context.copier.propose(
            &copy_context.history,
            copy_context.copy_row_budget + tree.token_ids.len(),
        );
        copied_row_count = rows.add_copied_chain(&copied);
    }
    let verification = pair
        .target
        .verify_draft_tree(
            &rows,
            round.fed_token_count,
            decoder_state,
            &pair.drafter_configuration.target_tap_layer_ids,
        )
        .expect("the tree verifies");
    let segment_started_at = record_segment(1, segment_started_at);
    // A row's pick takes the position after the row's own.
    let mut pick_positions = Vec::with_capacity(rows.row_count());
    for row_position in rows.positions(round.fed_token_count as i32) {
        pick_positions.push(row_position as u64 + 1);
    }
    let picks = target_picks(pair, cell, &verification.per_row_logits, &pick_positions);
    let segment_started_at = record_segment(2, segment_started_at);
    let kept = rows.kept_chain(&picks);
    if let Some(copy_context) = copy_context {
        let mut landed = 0;
        for kept_row in &kept.rows {
            if *kept_row >= first_copied_row {
                landed += 1;
            }
        }
        copy_context.copier.observe(copied_row_count, landed);
        adapt_copy_row_budget(copy_context, copied_row_count, landed);
    }
    round.pending_tap_rows = kept_tap_rows(pair, &verification.hidden_state_taps, &kept.rows);
    pair.target
        .commit_draft_tree(decoder_state, verification, &kept.rows)
        .expect("the kept chain commits");
    record_segment(3, segment_started_at);
    let mut emitted = Vec::with_capacity(kept.rows.len());
    for kept_row in &kept.rows[1..] {
        emitted.push(rows.token_ids[*kept_row]);
    }
    emitted.push(kept.next_token_id);
    round.fed_token_count += kept.rows.len() as u32;
    round.last_token_id = kept.next_token_id;
    emitted
}

/// Wall time an unsynced round spends in each segment, in microseconds:
/// drafting through the lattice readback, the verify's graph build, the
/// wait for the picks, and the commit's graph build. While the first and
/// third wait on the GPU, the second overlaps it; the fourth leaves it idle.
static SEGMENT_MICROSECONDS: [std::sync::atomic::AtomicU64; 4] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];
const SEGMENT_LABELS: [&str; 4] = [
    "draft + lattice",
    "verify build",
    "picks wait",
    "commit build",
];

fn record_segment(segment_index: usize, segment_started_at: Instant) -> Instant {
    SEGMENT_MICROSECONDS[segment_index].fetch_add(
        segment_started_at.elapsed().as_micros() as u64,
        std::sync::atomic::Ordering::Relaxed,
    );
    Instant::now()
}

/// Prints each segment's milliseconds a round and clears the totals.
fn print_segments(round_count: usize) {
    let mut line = String::new();
    for (label, total) in SEGMENT_LABELS.iter().zip(SEGMENT_MICROSECONDS.iter()) {
        let microseconds = total.swap(0, std::sync::atomic::Ordering::Relaxed);
        line.push_str(&format!(
            " {label} {:.1} ms,",
            microseconds as f64 / 1000.0 / round_count as f64
        ));
    }
    eprintln!("[dflash2-tree-rounds] a round, unsynced:{line}");
}

/// TensorFold's rule: double the copied rows while copies land whole, halve
/// them (not below the first budget) when fewer than half land.
fn adapt_copy_row_budget(copy_context: &mut CopyContext, copied_row_count: usize, landed: usize) {
    if copied_row_count == 0 {
        return;
    }
    if landed == copied_row_count {
        copy_context.copy_row_budget = (copy_context.copy_row_budget * 2).min(MAXIMUM_COPY_ROWS);
    } else if 2 * landed < copied_row_count {
        copy_context.copy_row_budget = (copy_context.copy_row_budget / 2).max(FIRST_COPY_ROWS);
    }
}

/// The drafter's next input: every tap's kept rows, in chain order, joined.
pub(super) fn kept_tap_rows(
    pair: &JourneyPair,
    layer_taps: &[MlxArray],
    kept_rows: &[usize],
) -> MlxArray {
    let runtime = pair.target.runtime();
    let mut kept_row_ids = Vec::with_capacity(kept_rows.len());
    for kept_row in kept_rows {
        kept_row_ids.push(*kept_row as i32);
    }
    let kept_row_array = runtime
        .array_from_i32(&kept_row_ids, &[kept_rows.len() as i32])
        .expect("the kept rows build");
    let mut kept_taps = Vec::with_capacity(layer_taps.len());
    for layer_tap in layer_taps {
        kept_taps.push(
            runtime
                .take_axis(layer_tap, &kept_row_array, 1)
                .expect("the kept tap rows gather"),
        );
    }
    let references: Vec<&MlxArray> = kept_taps.iter().collect();
    runtime
        .concatenate_axis(&references, -1)
        .expect("the taps join")
}
