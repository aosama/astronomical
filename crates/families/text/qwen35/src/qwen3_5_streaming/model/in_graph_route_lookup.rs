//! The host step that looks one layer's reply route up in its table, from
//! inside the token's graph.
//!
//! The step runs on MLX's CPU stream thread and reaches the model through a
//! pointer, which is the one piece of unsafe code in in-graph serving and
//! the reason this module is apart from `in_graph_table_serving`. Two facts
//! make it sound, and both are kept by `forward_resolving_expert_routes`:
//!
//! 1. **The model is there.** A lookup uses its pointer only while its
//!    forward's window is open. The window is opened and closed by the
//!    forward that built the lookup, which holds `&self` on the model for
//!    all that time, so the model is alive and has not moved. A lookup that
//!    runs at any other time (a graph kept past its forward) refuses.
//! 2. **Nothing else is using it.** The model is not `Sync`; its tables sit
//!    behind a `RefCell`. A lookup runs only during an MLX evaluation, and
//!    while the window is open every evaluation is a blocking call made by
//!    the thread that holds `&self`: an in-graph forward submits no layer
//!    group to run while it builds the next (`build_decoder_layers`). MLX
//!    returns from a blocking evaluation only after every step it queued
//!    has run, so the model's thread and MLX's never touch the model at the
//!    same moment, and MLX's queue lock orders what each sees.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use graph_host_step::{HostStep, HostStepFailed, HostStepInputs};

use super::Qwen3_5Model;
use crate::qwen3_5_streaming::expert_streaming::{TableReadThrough, TableRefusal};

/// The span of one in-graph forward, shared by the forward and its lookups.
#[derive(Debug, Default)]
pub struct InGraphForwardWindow {
    is_open: AtomicBool,
    /// Set by the first lookup that could not serve its layer. The token's
    /// outputs are wrong from that layer on and the token is served again,
    /// so the lookups after it read nothing from storage for routes that
    /// were never the token's.
    was_refused: AtomicBool,
}

impl InGraphForwardWindow {
    pub(super) fn open(&self) {
        self.was_refused.store(false, Ordering::SeqCst);
        self.is_open.store(true, Ordering::SeqCst);
    }

    pub(super) fn close(&self) {
        self.is_open.store(false, Ordering::SeqCst);
    }

    pub(super) fn is_open(&self) -> bool {
        self.is_open.load(Ordering::SeqCst)
    }

    pub(super) fn was_refused(&self) -> bool {
        self.was_refused.load(Ordering::SeqCst)
    }

    pub(super) fn mark_refused(&self) {
        self.was_refused.store(true, Ordering::SeqCst);
    }
}

/// What one layer's lookup did, read by the model's thread once the token's
/// evaluation has returned.
#[derive(Debug, Default)]
pub enum InGraphLayerOutcome {
    /// The evaluation has not reached the lookup.
    #[default]
    NotRun,
    Served(TableReadThrough),
    /// The table could not serve the route from inside the evaluation, or
    /// (`None`) a lookup before this one had already refused.
    Refused(Option<TableRefusal>),
    /// A read from storage failed.
    Failed(String),
}

/// One layer's outcome, written on MLX's thread and read on the model's.
#[derive(Debug, Default)]
pub struct SharedLayerOutcome(Mutex<InGraphLayerOutcome>);

impl SharedLayerOutcome {
    /// A lookup that panicked while writing still left a whole value behind.
    pub(super) fn lock(&self) -> MutexGuard<'_, InGraphLayerOutcome> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// The model as a lookup reaches it from MLX's thread.
struct ModelDuringForward(*const Qwen3_5Model);

// SAFETY: the pointer crosses to MLX's CPU stream thread and is used there
// only under the two conditions in this module's documentation: while the
// forward that built the lookup holds the model, and while that forward's
// thread is blocked in the evaluation that runs the lookup.
unsafe impl Send for ModelDuringForward {}

/// What a lookup knows of the table its graph multiplies from.
#[derive(Clone, Copy, Debug)]
pub struct InGraphLookupTable {
    /// The most slots the reply's plan gives the table.
    pub(super) warm_slot_count: usize,
    /// How many slots the table had when the graph took its arrays; a table
    /// with another count now is not the one the graph reads.
    pub(super) slot_count_when_built: usize,
}

/// Turns one layer's routed expert ids into slots of its table, reading the
/// experts the table lacks into it first.
pub(super) struct InGraphRouteLookup {
    model: ModelDuringForward,
    window: Arc<InGraphForwardWindow>,
    layer_index: usize,
    lookup_table: InGraphLookupTable,
    outcome: Arc<SharedLayerOutcome>,
}

impl InGraphRouteLookup {
    pub(super) fn new(
        model: &Qwen3_5Model,
        window: Arc<InGraphForwardWindow>,
        layer_index: usize,
        lookup_table: InGraphLookupTable,
        outcome: Arc<SharedLayerOutcome>,
    ) -> Self {
        Self {
            model: ModelDuringForward(model),
            window,
            layer_index,
            lookup_table,
            outcome,
        }
    }

    /// The refusal of a lookup that must not use its model pointer: its
    /// forward is over, or a lookup before it in the same forward refused.
    fn refusal_before_reaching_the_model(&self) -> Option<InGraphLayerOutcome> {
        (!self.window.is_open() || self.window.was_refused())
            .then_some(InGraphLayerOutcome::Refused(None))
    }

    fn look_up(
        &self,
        inputs: HostStepInputs<'_>,
        table_slots: &mut [[u8; 4]],
    ) -> InGraphLayerOutcome {
        if let Some(refusal) = self.refusal_before_reaching_the_model() {
            return refusal;
        }
        let Some(route_bytes) = inputs.bytes(0) else {
            return InGraphLayerOutcome::Failed("the lookup was given no route".to_owned());
        };
        let (routed_expert_assignments, _) = route_bytes.as_chunks::<4>();
        // SAFETY: the window is open, so the forward that built this lookup
        // holds the model and is blocked in the evaluation running it; see
        // the module documentation.
        let model = unsafe { &*self.model.0 };
        match model.read_reply_route_through_table(
            self.layer_index,
            self.lookup_table,
            routed_expert_assignments,
            table_slots,
        ) {
            Ok(Ok(table_read_through)) => InGraphLayerOutcome::Served(table_read_through),
            Ok(Err(table_refusal)) => InGraphLayerOutcome::Refused(Some(table_refusal)),
            Err(read_error) => InGraphLayerOutcome::Failed(read_error.to_string()),
        }
    }
}

impl HostStep for InGraphRouteLookup {
    /// Always reports success to MLX: a route the table cannot serve is the
    /// model's to deal with once the evaluation returns, as an optimistic
    /// token's missing experts are, and failing the evaluation would hide
    /// which layer it was.
    fn run(&mut self, inputs: HostStepInputs<'_>, output: &mut [u8]) -> Result<(), HostStepFailed> {
        let (table_slots, _) = output.as_chunks_mut::<4>();
        let lookup_started_at = std::time::Instant::now();
        let mut outcome = self.look_up(inputs, table_slots);
        if let InGraphLayerOutcome::Served(table_read_through) = &mut outcome {
            table_read_through.lookup_elapsed = lookup_started_at.elapsed();
        }
        if !matches!(outcome, InGraphLayerOutcome::Served(_)) {
            self.window.mark_refused();
            // The GPU multiplies from whatever slots it is handed. Slot zero
            // exists in every table, so a token that is about to be served
            // again reads inside its table meanwhile.
            table_slots.fill([0; 4]);
        }
        *self.outcome.lock() = outcome;
        Ok(())
    }
}

#[cfg(test)]
#[path = "in_graph_route_lookup_tests.rs"]
pub(super) mod in_graph_route_lookup_tests;
