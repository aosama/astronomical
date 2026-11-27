//! One expert layer's routed input, passed by reference through every way of
//! serving the layer, so each serving path names it once instead of
//! restating the same eight arguments.

use mlx_bridge::MlxArray;

use super::super::expert_streaming::expert_pager::Qwen3_5ExpertPager;
use super::feed_forward_weights::Qwen3_5MoEFeedForwardWeights;

/// Everything the serving paths read about one layer's routed forward. It
/// borrows from the forward that built it, which keeps it free to copy.
#[derive(Clone, Copy)]
pub(super) struct RoutedLayerInput<'forward> {
    pub(super) expert_pager: &'forward Qwen3_5ExpertPager,
    pub(super) layer_index: usize,
    pub(super) token_count: i32,
    pub(super) selected_indices: &'forward MlxArray,
    pub(super) selected_scores: &'forward MlxArray,
    pub(super) mixture_of_experts_weights: &'forward Qwen3_5MoEFeedForwardWeights,
    pub(super) hidden_states: &'forward MlxArray,
    pub(super) should_use_compiled_elementwise_graphs: bool,
}
