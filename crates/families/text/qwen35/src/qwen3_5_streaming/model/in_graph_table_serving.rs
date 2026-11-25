//! Serving a reply token with each layer's route looked up inside the graph.
//!
//! The exact way ends the evaluation at every layer not held whole: the
//! route is copied to the host, missing experts are read into the table,
//! and the rest of the layer is built from the answer. The optimistic way
//! keeps the route on the GPU and is right only when nothing was missing.
//! This way keeps the token one evaluation and is right when something was:
//! a host step between the router and the experts (`graph-host-step`) turns
//! the routed expert ids into table slots, reading the missing experts into
//! the table on the way.
//!
//! Measured through the engine's reply loop, 10,000 tokens in and up to 500
//! out, each case its own process (`cargo measure-in-graph-route-lookup`):
//!
//! | Model, MLX limit | Temperature | Routes on the host | Looked up in the graph |
//! |---|---|---|---|
//! | Qwen3.8-Flash-Next 4-bit, 34 GB | 1.0 | 17.8 tok/s | 19.3 tok/s |
//! | Qwen3.8-Flash-Next 4-bit, 34 GB | 0.7 | 17.8 tok/s | 20.2 tok/s |
//! | Ornith 35B-A3B OptiQ 4-bit, 16 GB | 1.0 | 48.6 tok/s | 56.0 tok/s |
//! | Ornith 35B-A3B OptiQ 4-bit, 16 GB | 0.7 | 49.0 tok/s | 57.0 tok/s |
//!
//! The larger model gains less because it is bound by storage: about 40
//! percent of its reply is spent reading experts, which a lookup does not
//! shorten. A greedy reply is the same tokens either way
//! (`cargo test-in-graph-route-lookup`).
//!
//! A lookup refuses what only the model's own thread can do, which is
//! growing a table; the token is then served again the exact way. So tables
//! are grown between tokens, before a lookup can meet one that is full
//! (`grow_tables_ahead_of_lookups`), and in those measurements no token was
//! refused. Should refusals come anyway, the tokens after one are served
//! exactly for a while, since each refusal costs a forward.
//!
//! `in_graph_route_lookup` holds the step and the argument for why it may
//! reach the model from MLX's thread.

use std::cell::{Cell, RefCell};
use std::sync::Arc;

use graph_host_step::{HostStepOutput, append_host_step};
use mlx_bridge::{MlxArray, MlxDtype};

use super::Qwen3_5Model;
use super::feed_forward_weights::Qwen3_5MoEFeedForwardWeights;
use super::in_graph_route_lookup::{
    InGraphForwardWindow, InGraphLayerOutcome, InGraphLookupTable, InGraphRouteLookup,
    SharedLayerOutcome,
};
use super::paged_execution::PageSlots;
use crate::qwen3_5_streaming::expert_streaming::{TableReadThrough, TableRefusal};
use crate::{PerformanceAttribution, should_commit_mandatory_routed_page};
use text_generation::execution_error::TextExecutionError;

/// The most reply tokens served exactly after a refusal before the next
/// in-graph attempt. A refused token costs one wasted forward, so at this
/// spacing a table that never stops refusing costs about three percent.
const MAXIMUM_EXACT_TOKENS_AFTER_A_REFUSAL: u32 = 32;

/// One layer of an in-graph forward, kept until the token's evaluation
/// returns and its lookup's outcome can be read.
#[derive(Debug)]
pub(super) struct InGraphLayerServing {
    pub(super) layer_index: usize,
    pub(super) assignment_count: usize,
    pub(super) table_payload_bytes: u64,
    pub(super) expert_capacity: usize,
    pub(super) outcome: Arc<SharedLayerOutcome>,
}

/// How many reply tokens went which way since the model loaded.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct InGraphServingCounts {
    /// Tokens attempted with in-graph lookups.
    pub attempted_token_count: u64,
    /// Attempts a lookup refused, which were then served exactly.
    pub refused_token_count: u64,
    /// Layers a lookup served, and those of them that read from storage.
    pub served_layer_count: u64,
    pub fetching_layer_count: u64,
    /// Refused tokens by what their first refusing layer needed: a larger
    /// table, or more seats than its table would give.
    pub would_grow_refusal_count: u64,
    pub cannot_seat_refusal_count: u64,
}

/// A model's in-graph serving: whether it is on, the forward in progress,
/// and what recent tokens say about attempting the next one.
#[derive(Debug)]
pub struct InGraphTableServing {
    /// Off until `set_in_graph_table_serving` turns it on with MLX's fast
    /// synch, which the engine does when it loads a model that streams
    /// experts; a model loaded directly, as a contract loads one, serves
    /// the exact way unless it asks.
    is_enabled: Cell<bool>,
    /// Off only while a contract makes lookups refuse on purpose, to prove
    /// a refused token is served again correctly.
    grows_tables_ahead_of_lookups: Cell<bool>,
    window: Arc<InGraphForwardWindow>,
    pub(super) layer_servings: RefCell<Vec<InGraphLayerServing>>,
    /// Reply tokens still to serve exactly before the next attempt, and how
    /// many the next refusal will set that to.
    exact_tokens_before_next_attempt: Cell<u32>,
    exact_tokens_after_next_refusal: Cell<u32>,
    pub(super) counts: Cell<InGraphServingCounts>,
}

impl Default for InGraphTableServing {
    fn default() -> Self {
        Self {
            is_enabled: Cell::new(false),
            grows_tables_ahead_of_lookups: Cell::new(true),
            window: Arc::default(),
            layer_servings: RefCell::default(),
            exact_tokens_before_next_attempt: Cell::new(0),
            // The first refusal is followed by one exact token.
            exact_tokens_after_next_refusal: Cell::new(1),
            counts: Cell::default(),
        }
    }
}

impl InGraphTableServing {
    /// Whether the forward being built looks its routes up in the graph.
    pub(super) fn is_serving_forward(&self) -> bool {
        self.window.is_open()
    }

    /// Whether the next reply token waits out a recent refusal, in which
    /// case it is served exactly and counts against the wait.
    fn waits_out_a_refusal(&self) -> bool {
        let exact_tokens_left = self.exact_tokens_before_next_attempt.get();
        if exact_tokens_left == 0 {
            return false;
        }
        self.exact_tokens_before_next_attempt
            .set(exact_tokens_left - 1);
        true
    }

    /// A refused token is followed by exact tokens, twice as many after each
    /// refusal in a row up to `MAXIMUM_EXACT_TOKENS_AFTER_A_REFUSAL`.
    fn space_out_attempts_after_a_refusal(&self) {
        let exact_token_count = self.exact_tokens_after_next_refusal.get();
        self.exact_tokens_before_next_attempt.set(exact_token_count);
        self.exact_tokens_after_next_refusal.set(
            exact_token_count
                .saturating_mul(2)
                .min(MAXIMUM_EXACT_TOKENS_AFTER_A_REFUSAL),
        );
    }

    /// Decides, once a forward has been evaluated, whether a lookup refused
    /// one of its layers, so the token must be served again. A refusal also
    /// spaces out the next attempts; a forward the lookups served resets the
    /// spacing. A forward that is not in-graph (an exact one, the re-run of
    /// a refused token included) has no lookups and was not refused.
    ///
    /// # Errors
    /// Fails when a lookup of a forward nobody refused never ran: its layer
    /// then multiplied from slots that were never written.
    fn conclude_forward(&self) -> Result<bool, TextExecutionError> {
        if !self.forward_was_refused() {
            self.require_every_lookup_served()?;
            if !self.layer_servings.borrow().is_empty() {
                self.forget_refusals();
            }
            return Ok(false);
        }
        let mut counts = self.counts.get();
        for layer_serving in self.layer_servings.borrow().iter() {
            match &*layer_serving.outcome.lock() {
                InGraphLayerOutcome::Failed(description) => {
                    // The exact way reads the same experts next and reports
                    // the failure as a request error if it persists.
                    tracing::warn!(
                        layer_index = layer_serving.layer_index,
                        description,
                        "an in-graph lookup could not read an expert"
                    );
                }
                InGraphLayerOutcome::Refused(Some(TableRefusal::WouldGrow)) => {
                    counts.would_grow_refusal_count += 1;
                }
                InGraphLayerOutcome::Refused(Some(TableRefusal::CannotSeatEveryExpert)) => {
                    counts.cannot_seat_refusal_count += 1;
                }
                _ => {}
            }
        }
        self.layer_servings.borrow_mut().clear();
        self.space_out_attempts_after_a_refusal();
        counts.refused_token_count += 1;
        self.counts.set(counts);
        Ok(true)
    }

    /// Every lookup of a forward that nobody refused must have run and
    /// served its layer.
    fn require_every_lookup_served(&self) -> Result<(), TextExecutionError> {
        for layer_serving in self.layer_servings.borrow().iter() {
            if !matches!(
                *layer_serving.outcome.lock(),
                InGraphLayerOutcome::Served(_)
            ) {
                return Err(TextExecutionError::InvalidInput {
                    description: "an in-graph lookup did not run in its token's evaluation",
                });
            }
        }
        Ok(())
    }

    /// A served token ends a run of refusals: the next one waits one token.
    fn forget_refusals(&self) {
        self.exact_tokens_after_next_refusal.set(1);
    }

    /// Closes the window and forgets the forward's layers. An evaluated
    /// forward has already recorded or discarded them; one whose build or
    /// evaluation failed has not, and left behind they would be concluded
    /// with the next forward, failing an exact one for lookups it never had.
    fn end_forward(&self) {
        self.window.close();
        self.layer_servings.borrow_mut().clear();
    }

    /// Whether a lookup of the forward in progress has refused its layer.
    pub(super) fn forward_was_refused(&self) -> bool {
        self.window.is_open() && self.window.was_refused()
    }

    pub fn counts(&self) -> InGraphServingCounts {
        self.counts.get()
    }
}

impl Qwen3_5Model {
    /// This model's in-graph serving. It is kept with the forward's other
    /// speculative state, which is checked in the same place.
    pub fn in_graph_table_serving(&self) -> &InGraphTableServing {
        &self
            .paged_forward_missing_route_collector
            .in_graph_table_serving
    }

    /// Turns in-graph serving on or off, and MLX's fast synch with it: a
    /// lookup inside the graph only pays under fast synch, and nothing else
    /// in a streamed reply crosses from the GPU to the host inside a graph.
    pub fn set_in_graph_table_serving(&self, is_enabled: bool) {
        self.in_graph_table_serving().is_enabled.set(is_enabled);
        graph_host_step::set_fast_synch(is_enabled);
    }

    /// Whether a layer that stops at the host reads its missing experts
    /// straight into its table: the direct read is on, and no in-graph
    /// lookup of this forward has refused. After a refusal the layer is
    /// routing from wrong hidden states and the token is about to be served
    /// again, so its experts are not worth a slot.
    pub(super) fn reads_host_routes_straight_into_tables(&self) -> bool {
        self.direct_expert_fetching.is_enabled.get()
            && !self.in_graph_table_serving().forward_was_refused()
    }

    /// How many layers a one-token forward runs before it submits the
    /// finished group. An in-graph forward submits none ahead of its one
    /// evaluation: a group running while the next is built would run its
    /// lookups while this thread reads the same tables
    /// (`in_graph_route_lookup`).
    pub(super) fn reply_graph_submission_layer_interval(&self, pacing_token_count: i32) -> usize {
        if self.in_graph_table_serving().is_serving_forward() {
            return 0;
        }
        self.graph_submission_layer_interval(pacing_token_count, true)
    }

    /// Stops or resumes growing tables between tokens, for a contract that
    /// needs lookups to refuse.
    #[doc(hidden)]
    pub fn set_tables_grown_ahead_of_lookups_for_tests(&self, is_enabled: bool) {
        self.in_graph_table_serving()
            .grows_tables_ahead_of_lookups
            .set(is_enabled);
    }

    /// Whether the next forward is attempted with in-graph lookups: a single
    /// reply token, on a model that can read experts straight into its
    /// tables, with at least one layer a lookup can serve and no recent
    /// refusal still being waited out.
    pub(super) fn should_serve_in_graph(&self, token_count: i32) -> bool {
        let in_graph_serving = self.in_graph_table_serving();
        if token_count != 1
            || !in_graph_serving.is_enabled.get()
            || self.brings_every_route_to_the_host.get()
            || !self.direct_expert_fetching.is_enabled.get()
            || self.direct_expert_fetching.slot_filler.is_none()
            || !self.has_a_layer_a_lookup_can_serve()
        {
            return false;
        }
        !in_graph_serving.waits_out_a_refusal()
    }

    /// The tables and the pager of a model that streams experts.
    fn expert_tables_and_pager(
        &self,
    ) -> Option<(
        &RefCell<crate::qwen3_5_streaming::expert_streaming::AstroExpertCache>,
        &crate::qwen3_5_streaming::expert_streaming::expert_pager::Qwen3_5ExpertPager,
    )> {
        Some((
            self.astro_expert_cache.as_ref()?,
            self.expert_pager.as_ref()?,
        ))
    }

    fn has_a_layer_a_lookup_can_serve(&self) -> bool {
        let Some((astro_expert_cache, expert_pager)) = self.expert_tables_and_pager() else {
            return false;
        };
        let astro_expert_cache = astro_expert_cache.borrow();
        for (layer_index, layer_plan) in expert_pager.layer_plans().iter().enumerate() {
            if self.keeps_reply_experts_on(layer_index)
                && astro_expert_cache
                    .partly_held_table(layer_index, layer_plan.expert_capacity)
                    .is_some()
            {
                return true;
            }
        }
        false
    }

    /// The rule a fetch from the model's thread applies: a layer the reply's
    /// plan keeps nothing new on is not refilled behind it.
    fn keeps_reply_experts_on(&self, layer_index: usize) -> bool {
        self.hot_expert_warm_slot_count() > 0
            && should_commit_mandatory_routed_page(
                1,
                true,
                self.expert_layer_state(layer_index),
                false,
            )
    }

    pub(super) fn begin_in_graph_forward(&self) {
        let in_graph_serving = self.in_graph_table_serving();
        if in_graph_serving.grows_tables_ahead_of_lookups.get() {
            self.grow_tables_ahead_of_lookups();
        }
        in_graph_serving.layer_servings.borrow_mut().clear();
        in_graph_serving.window.open();
        let mut counts = in_graph_serving.counts.get();
        counts.attempted_token_count += 1;
        in_graph_serving.counts.set(counts);
    }

    /// Grows, before the token is built, every table a lookup would
    /// otherwise refuse to fetch into for want of free slots. A table that
    /// cannot be grown is left to its lookup, which refuses, and the exact
    /// way then reports whatever is wrong with it.
    fn grow_tables_ahead_of_lookups(&self) {
        let Some((astro_expert_cache, expert_pager)) = self.expert_tables_and_pager() else {
            return;
        };
        let warm_slot_count = self.hot_expert_warm_slot_count();
        let experts_per_token =
            usize::try_from(self.config.experts_per_token()).unwrap_or(usize::MAX);
        for (layer_index, layer_plan) in expert_pager.layer_plans().iter().enumerate() {
            if !self.keeps_reply_experts_on(layer_index) {
                continue;
            }
            if let Err(growth_error) = astro_expert_cache.borrow_mut().grow_table_ahead_of_lookups(
                &self.runtime,
                layer_index,
                warm_slot_count.min(layer_plan.expert_capacity),
                experts_per_token,
            ) {
                tracing::warn!(
                    layer_index,
                    %growth_error,
                    "a table could not be grown ahead of its lookup"
                );
            }
        }
    }

    /// Closes the forward's window, after which a lookup that has not run
    /// refuses. Called whether the forward was evaluated or failed.
    pub(super) fn end_in_graph_forward(&self) {
        self.in_graph_table_serving().end_forward();
    }

    /// Serves one layer of an in-graph forward from its table, with a host
    /// step naming the slots. `Ok(None)` leaves the layer to the ways that
    /// were there before: this forward is not in-graph, the table holds the
    /// whole layer (no lookup is needed) or none of it, or the layer keeps
    /// no new experts. Such a layer may still stop at the host; the lookups
    /// before it have run by then, and those after it run later.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn dispatch_in_graph_table_page(
        &self,
        layer_index: usize,
        selected_indices: &MlxArray,
        selected_scores: &MlxArray,
        hidden_states: &MlxArray,
        mixture_of_experts_weights: &Qwen3_5MoEFeedForwardWeights,
        should_use_compiled_elementwise_graphs: bool,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Option<MlxArray>, TextExecutionError> {
        let in_graph_serving = self.in_graph_table_serving();
        if !in_graph_serving.is_serving_forward()
            || selected_indices.dtype() != MlxDtype::UInt32
            || !self.keeps_reply_experts_on(layer_index)
        {
            return Ok(None);
        }
        let Some((astro_expert_cache, expert_pager)) = self.expert_tables_and_pager() else {
            return Ok(None);
        };
        let expert_capacity = expert_pager.layer_plan(layer_index)?.expert_capacity;
        let Some(partly_held_table) = astro_expert_cache
            .borrow()
            .partly_held_table(layer_index, expert_capacity)
        else {
            return Ok(None);
        };
        // An in-graph forward is one reply token.
        if !partly_held_table
            .weights
            .multiplies_complete_page_without_transient_decode(
                1,
                self.fused_jangtq2_gathered_projection_kernel.is_some(),
            )
        {
            return Ok(None);
        }
        let outcome = Arc::new(SharedLayerOutcome::default());
        let table_slots = append_host_step(
            &[selected_indices],
            HostStepOutput {
                shape: &selected_indices.shape(),
                dtype: MlxDtype::UInt32,
            },
            Box::new(InGraphRouteLookup::new(
                self,
                Arc::clone(&in_graph_serving.window),
                layer_index,
                InGraphLookupTable {
                    warm_slot_count: self.hot_expert_warm_slot_count().min(expert_capacity),
                    slot_count_when_built: partly_held_table.slot_count,
                },
                Arc::clone(&outcome),
            )),
        )?;
        in_graph_serving
            .layer_servings
            .borrow_mut()
            .push(InGraphLayerServing {
                layer_index,
                assignment_count: selected_indices.element_count(),
                table_payload_bytes: partly_held_table.payload_bytes,
                expert_capacity,
                outcome,
            });
        // The lookup records the route's demand itself, before it plans
        // slots, as a route resolved on the host does.
        let layer_output = self.forward_moe_from_page_slots(
            PageSlots::LookedUp(&table_slots),
            selected_scores,
            &partly_held_table.weights,
            hidden_states,
            mixture_of_experts_weights,
            should_use_compiled_elementwise_graphs,
            performance_attribution,
        )?;
        Ok(Some(layer_output))
    }

    /// The lookup itself, called by `InGraphRouteLookup` on MLX's thread
    /// while this model's own thread waits in the evaluation.
    pub(super) fn read_reply_route_through_table(
        &self,
        layer_index: usize,
        lookup_table: InGraphLookupTable,
        routed_expert_assignments: &[[u8; 4]],
        table_slots: &mut [[u8; 4]],
    ) -> Result<Result<TableReadThrough, TableRefusal>, TextExecutionError> {
        let (Some(astro_expert_cache), Some(slot_filler), Some(expert_pager)) = (
            self.astro_expert_cache.as_ref(),
            self.direct_expert_fetching.slot_filler.as_ref(),
            self.expert_pager.as_ref(),
        ) else {
            return Ok(Err(TableRefusal::TableUnavailable));
        };
        let layer_plan = expert_pager.layer_plan(layer_index)?;
        // The model's thread holds no borrow of the tables across an
        // evaluation; if that ever stops being true the lookup refuses
        // instead of panicking inside MLX.
        let Ok(mut astro_expert_cache) = astro_expert_cache.try_borrow_mut() else {
            return Ok(Err(TableRefusal::TableUnavailable));
        };
        // The graph multiplies from the table arrays it was built with. A
        // table replaced since (grown by a layer that stopped at the host)
        // would take this lookup's experts where that graph does not read.
        if astro_expert_cache.table_slot_count(layer_index)
            != Some(lookup_table.slot_count_when_built)
        {
            return Ok(Err(TableRefusal::TableUnavailable));
        }
        Ok(astro_expert_cache.read_route_through_table(
            &self.runtime,
            layer_index,
            layer_plan,
            routed_expert_assignments,
            lookup_table.warm_slot_count,
            slot_filler,
            table_slots,
        )?)
    }

    /// Whether a lookup of the forward just evaluated refused its layer, so
    /// the token must be served again.
    ///
    /// # Errors
    /// Fails when a lookup of a forward nobody refused never ran.
    pub(super) fn in_graph_forward_was_refused(&self) -> Result<bool, TextExecutionError> {
        self.in_graph_table_serving().conclude_forward()
    }
}

#[cfg(test)]
#[path = "in_graph_table_serving_tests.rs"]
mod in_graph_table_serving_tests;
