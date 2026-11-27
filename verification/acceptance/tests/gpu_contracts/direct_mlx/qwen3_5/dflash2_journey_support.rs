//! Small helpers for the DFlash2 journeys on a real target and drafter pair:
//! the target's forwards with the drafter's taps, as a drafting engine runs
//! them on the one text model, and host reads.

use mlx_bridge::{MlxArray, MlxDtype};
use model_serving::qwen3_5_text::model::drafted_rows_verification::DraftedRowsVerification;
use model_serving::{PerformanceAttribution, Qwen3_5TextModel, RequestDecoderStateStack};
use runtime_integration::MlxRuntime;

/// Which rows of a target forward the head scores: the last alone, for a
/// prompt whose one read is the first reply token, or every row.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum TargetLogitRows {
    LastRowOnly,
    EveryRow,
}

/// One target forward as a journey reads it: float32 logits
/// `[1, rows, vocabulary]` for the rows asked, and one `[1, rows, hidden]`
/// array a tap, in the drafter's tap order.
pub(crate) struct TargetForward {
    pub(crate) logits: MlxArray,
    pub(crate) hidden_state_taps: Vec<MlxArray>,
}

/// Forwards `token_ids` from `first_position`, keeping the rows after each
/// of `tap_layer_ids`, the way a drafting engine prefills its prompt.
pub(crate) fn forward_with_taps(
    target: &Qwen3_5TextModel,
    token_ids: &[u32],
    first_position: u32,
    decoder_state: &mut RequestDecoderStateStack,
    tap_layer_ids: &[i32],
    logit_rows: TargetLogitRows,
) -> TargetForward {
    if !tap_layer_ids.is_empty() {
        target.begin_draft_tap_capture(tap_layer_ids, token_ids.len());
    }
    let forward_output = target
        .forward_chunk_with_pre_final_normalization_hidden_states(
            token_ids,
            first_position,
            decoder_state,
        )
        .expect("the target forward runs");
    let final_rows = forward_output.pre_final_normalization_hidden_states();
    let scored_rows = match logit_rows {
        TargetLogitRows::EveryRow => final_rows.retain().expect("the final rows handle"),
        TargetLogitRows::LastRowOnly => last_row(target.runtime(), final_rows),
    };
    let logits = target
        .normalized_vocabulary_logits_for_every_row(&scored_rows)
        .expect("the head scores the rows");
    TargetForward {
        logits,
        hidden_state_taps: taken_taps(target, tap_layer_ids),
    }
}

/// A drafted block verified in one pass, and the rows after its taps.
pub(crate) struct VerifiedBlock {
    pub(crate) verification: DraftedRowsVerification,
    pub(crate) hidden_state_taps: Vec<MlxArray>,
}

/// Verifies `token_ids` (the last chosen token, then the drafts) from
/// `first_position`, keeping the rows after each of `tap_layer_ids`.
pub(crate) fn verify_block_with_taps(
    target: &Qwen3_5TextModel,
    token_ids: &[u32],
    first_position: u32,
    decoder_state: &mut RequestDecoderStateStack,
    tap_layer_ids: &[i32],
) -> VerifiedBlock {
    if !tap_layer_ids.is_empty() {
        target.begin_draft_tap_capture(tap_layer_ids, token_ids.len());
    }
    let verification = target
        .verify_drafted_rows(
            token_ids,
            first_position,
            decoder_state,
            &mut PerformanceAttribution::disabled(),
        )
        .expect("the target verifies the block");
    VerifiedBlock {
        verification,
        hidden_state_taps: taken_taps(target, tap_layer_ids),
    }
}

fn taken_taps(target: &Qwen3_5TextModel, tap_layer_ids: &[i32]) -> Vec<MlxArray> {
    if tap_layer_ids.is_empty() {
        return Vec::new();
    }
    let (_, hidden_state_taps) = target
        .take_draft_taps()
        .expect("the taps join")
        .expect("every requested tap was kept");
    hidden_state_taps
}

fn last_row(runtime: &MlxRuntime, rows: &MlxArray) -> MlxArray {
    let shape = rows.shape();
    runtime
        .slice(
            rows,
            &[0, shape[1] - 1, 0],
            &[1, shape[1], shape[2]],
            &[1, 1, 1],
        )
        .expect("the last row slices")
}

/// Every value of `array`, read to the host as float32.
pub(crate) fn host_values(runtime: &MlxRuntime, array: &MlxArray) -> Vec<f32> {
    let single_precision = runtime
        .astype(array, MlxDtype::Float32)
        .expect("values convert");
    runtime
        .array_to_vec_f32(&single_precision)
        .expect("values read back")
}

/// The first `row_count` rows of each tapped layer output [1, rows, H],
/// joined in tap order into the drafter's fusion input [1, row_count, taps x H].
pub(crate) fn tap_rows(runtime: &MlxRuntime, layer_taps: &[MlxArray], row_count: i32) -> MlxArray {
    let mut kept_rows = Vec::with_capacity(layer_taps.len());
    for layer_tap in layer_taps {
        let hidden_size = layer_tap.shape()[2];
        kept_rows.push(
            runtime
                .slice(
                    layer_tap,
                    &[0, 0, 0],
                    &[1, row_count, hidden_size],
                    &[1, 1, 1],
                )
                .expect("the kept tap rows slice"),
        );
    }
    let references: Vec<&MlxArray> = kept_rows.iter().collect();
    runtime
        .concatenate_axis(&references, -1)
        .expect("the taps join")
}

/// A seeded uniform stream for acceptance tests and leftover sampling
/// (splitmix64), so a run's verdicts are reproducible.
pub(crate) struct UniformStream {
    state: u64,
}

impl UniformStream {
    pub(crate) fn seeded(seed: u64) -> Self {
        Self { state: seed }
    }

    /// A value in [0, 1) with 53 random bits.
    pub(crate) fn next_uniform(&mut self) -> f64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut mixed = self.state;
        mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        mixed ^= mixed >> 31;
        (mixed >> 11) as f64 / (1_u64 << 53) as f64
    }
}

/// Romeo and Juliet words that, with the continuation instruction, make about
/// 2,000 prompt tokens. A dense target reads a prompt at about 250 tokens a
/// second, so a longer story only adds waiting to every decode that reads it.
const DENSE_STORY_PROMPT_WORD_COUNT: usize = 1_480;
/// About 10,000 prompt tokens, the size the serving throughput journey sends,
/// for a mixture-of-experts target, which reads its prompt far faster.
const SPARSE_STORY_PROMPT_WORD_COUNT: usize = 7_400;
const STORY_INSTRUCTION: &str =
    "Continue the supplied Romeo and Juliet story above for approximately 500 tokens.";

/// The story every DFlash journey continues: a Romeo and Juliet excerpt and
/// the continuation instruction, sized by the configured target's kind. It
/// is the one long-reply cell the journeys keep; a short chat question adds
/// nothing the story does not cover.
pub(crate) fn story_prompt() -> &'static str {
    static STORY_PROMPT: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    STORY_PROMPT.get_or_init(|| {
        let word_count = if configured_target_is_dense() {
            DENSE_STORY_PROMPT_WORD_COUNT
        } else {
            SPARSE_STORY_PROMPT_WORD_COUNT
        };
        let words: Vec<&str> =
            include_str!("../../../fixtures/model_metrics_10000_romeo_and_juliet_words.txt")
                .split_whitespace()
                .take(word_count)
                .collect();
        format!("{}\n\n{STORY_INSTRUCTION}", words.join(" "))
    })
}

/// The story as a chat turn at `temperature_thousandths`, at the engine's
/// own tree budget.
pub(crate) fn story_cell(
    label: &'static str,
    temperature_thousandths: u16,
) -> super::dflash2_kept_per_round::Cell {
    super::dflash2_kept_per_round::Cell {
        label,
        prompt: story_prompt(),
        is_raw_completion: false,
        temperature_thousandths,
        minimum_kept_per_round: 0.0,
        node_budget: super::dflash2_kept_per_round::ENGINE_NODE_BUDGET,
        children_per_node: super::dflash2_kept_per_round::ENGINE_CHILDREN_PER_NODE,
        keyed_seed: super::dflash2_tree_oracle::FIRST_KEYED_SEED,
    }
}

/// Whether the configured target is a dense model (no mixture of experts),
/// read from its validated configuration before any weight loads.
fn configured_target_is_dense() -> bool {
    let (target_directory, _, _) = super::dflash2_kept_per_round::pair_directories();
    model_serving::Qwen3_5ArtifactValidator::new()
        .validate(&target_directory, 4_096)
        .expect("the target artifact validates")
        .config()
        .feed_forward_architecture()
        == model_serving::Qwen3_5FeedForwardArchitecture::Dense
}
