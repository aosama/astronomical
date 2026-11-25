//! Looks a reply route up in a layer's table, reading through to storage
//! for the experts the table does not hold.
//!
//! This is the lookup a host step does inside the graph of a reply token
//! (`model/in_graph_table_serving.rs`): the GPU has just routed, the GPU's
//! next kernel multiplies from the table, and in between each routed expert
//! needs a slot. An expert the table holds has one. An expert it lacks is
//! given a slot and read into it by `direct_fetch`, by the same rules as a
//! fetch made from the model's own thread.
//!
//! What it cannot do is grow the table. Growing builds a larger tensor
//! through MLX, the evaluation in progress has already been handed the
//! present one, and a step inside an evaluation must not start another. So
//! where a fetch from the model's thread would grow the table, this refuses,
//! and the caller serves the token the way that can.

use runtime_integration::{MlxRuntime, MlxRuntimeError};

use super::AstroExpertCache;
use super::direct_fetch::TableFetchReport;
use super::insert_planning::{grow_table_within_budget, planned_table_growth};
use super::slot_writes::RetainedReferenceOk;
use crate::qwen3_5_streaming::expert_streaming::ExpertSlotStorageFiller;
use crate::qwen3_5_streaming::expert_streaming::expert_pager::Qwen3_5PagedExpertWeights;
use engine_core::expert_paging::QuantizedExpertLayerPlan;

/// Why a layer's table could not serve a route from inside an evaluation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TableRefusal {
    /// There is no table this lookup may use: the layer has none, the
    /// caller's slots do not match the route, the table was replaced after
    /// the graph took its arrays, or the model's thread holds the tables.
    TableUnavailable,
    /// A fetch from the model's thread would first grow the table.
    WouldGrow,
    /// The table cannot seat every missing expert: too few slots hold an
    /// expert this token does not use, or the policy admits fewer new
    /// experts a step than the route needs.
    CannotSeatEveryExpert,
}

/// Why a layer's table is not one a lookup inside the graph can serve
/// routes through, so the layer is served some other way.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TableNotLookedUp {
    /// The layer has no table.
    NoTable,
    /// The table has slots and holds no expert yet.
    HoldsNothing,
    /// The table holds the whole layer: its route is remapped on the GPU
    /// and needs no lookup.
    HoldsWholeLayer,
    /// The table's experts are in an encoding the reader cannot write
    /// straight into.
    CannotBeFilledFromStorage,
}

/// What one route's lookup found and did.
#[derive(Clone, Copy, Debug, Default)]
pub struct TableReadThrough {
    /// Distinct routed experts the table already held.
    pub retained_expert_count: usize,
    /// Distinct routed experts read from storage into the table.
    pub missing_expert_count: usize,
    /// The reads, when any expert was missing.
    pub table_fetch_report: Option<TableFetchReport>,
    /// How long the whole fetch took, planning and reads together.
    pub fetch_elapsed: std::time::Duration,
    /// How long the whole lookup took, from the route arriving to the slots
    /// being written; set by the host step that ran it.
    pub lookup_elapsed: std::time::Duration,
    /// Routed experts a previous token's prefetch had put in the table, and
    /// routed experts it had been asked for and did not deliver.
    pub prefetch_hit_count: u64,
    pub prefetch_miss_count: u64,
}

/// A table that holds part of its layer, as a graph multiplies from it
/// while a lookup names the slots.
#[derive(Debug)]
pub struct PartlyHeldTable {
    /// References to the table's own arrays: an expert read into a slot
    /// afterwards is in the storage these multiply from.
    pub weights: Qwen3_5PagedExpertWeights,
    pub payload_bytes: u64,
    /// How many slots those arrays have.
    pub slot_count: usize,
}

impl AstroExpertCache {
    /// The layer's table when a lookup can serve routes through it: it holds
    /// some of the layer's experts and not all of them, and experts can be
    /// read into it straight from storage. A table that holds its whole
    /// layer needs no lookup, and one that holds nothing has no slots yet.
    #[must_use]
    pub fn partly_held_table(
        &self,
        layer_index: usize,
        expert_capacity: usize,
    ) -> Option<PartlyHeldTable> {
        self.partly_held_table_or_why_not(layer_index, expert_capacity)
            .ok()
    }

    /// `partly_held_table`, saying why when there is none.
    pub fn partly_held_table_or_why_not(
        &self,
        layer_index: usize,
        expert_capacity: usize,
    ) -> Result<PartlyHeldTable, TableNotLookedUp> {
        let Some(Some(table)) = self.tables_by_layer.get(layer_index) else {
            return Err(TableNotLookedUp::NoTable);
        };
        if table.occupied_slot_count == 0 {
            return Err(TableNotLookedUp::HoldsNothing);
        }
        if table.occupied_slot_count >= expert_capacity {
            return Err(TableNotLookedUp::HoldsWholeLayer);
        }
        if !ExpertSlotStorageFiller::can_fill(&table.weights) {
            return Err(TableNotLookedUp::CannotBeFilledFromStorage);
        }
        Ok(PartlyHeldTable {
            weights: table.weights.retained_reference_ok(),
            payload_bytes: table.payload_bytes,
            slot_count: table.slot_count(),
        })
    }

    /// How many slots the layer's table has, or `None` without a table.
    #[must_use]
    pub fn table_slot_count(&self, layer_index: usize) -> Option<usize> {
        Some(
            self.tables_by_layer
                .get(layer_index)?
                .as_ref()?
                .slot_count(),
        )
    }

    /// Writes the table slot of every expert of `routed_expert_assignments`
    /// into `table_slots`, in the same order, after reading from storage the
    /// experts the layer's table lacked.
    ///
    /// Both lists are native-endian `u32` values as bytes, which is how a
    /// host step is handed a route and hands back the slots.
    ///
    /// Returns the refusal when the route cannot be served here, after which
    /// `table_slots` may be partly written and must not be used. Experts
    /// read before a refusal stay in the table; they were routed to.
    ///
    /// The caller must have evaluated every graph that reads this layer's
    /// table other than the one waiting on these slots.
    ///
    /// # Errors
    /// Fails when a read from storage fails; the slots planned for the
    /// experts it could not read are given up first.
    #[allow(clippy::too_many_arguments)]
    pub fn read_route_through_table(
        &mut self,
        runtime: &MlxRuntime,
        layer_index: usize,
        layer_plan: &QuantizedExpertLayerPlan,
        routed_expert_assignments: &[[u8; 4]],
        warm_slot_count: usize,
        slot_filler: &ExpertSlotStorageFiller,
        table_slots: &mut [[u8; 4]],
    ) -> Result<Result<TableReadThrough, TableRefusal>, MlxRuntimeError> {
        let Some(Some(table)) = self.tables_by_layer.get(layer_index) else {
            return Ok(Err(TableRefusal::TableUnavailable));
        };
        if table_slots.len() != routed_expert_assignments.len() {
            return Ok(Err(TableRefusal::TableUnavailable));
        }
        // A fetch grows a table only while it has fewer slots than the count
        // it is given. Capping that count at the slots the table has now
        // makes growth impossible here whatever the planner would decide,
        // which is what keeps this safe inside an evaluation; the refusal
        // below is what keeps it the same policy as a fetch that may grow.
        let slot_count_now = table.slot_count();
        let mut selected_expert_ids = Vec::with_capacity(routed_expert_assignments.len());
        for routed_expert_assignment in routed_expert_assignments {
            selected_expert_ids.push(u32::from_ne_bytes(*routed_expert_assignment) as usize);
        }
        let mut routed_expert_ids = selected_expert_ids.clone();
        routed_expert_ids.sort_unstable();
        routed_expert_ids.dedup();
        let route_coverage = self.routed_expert_coverage(layer_index, &routed_expert_ids);
        let missing_expert_count = route_coverage.missing_expert_ids.len();
        if self.would_grow_table_for(layer_index, missing_expert_count, warm_slot_count) {
            return Ok(Err(TableRefusal::WouldGrow));
        }
        // What a route resolved on the host records, in the same order: its
        // demand before the fetch, so the experts it asks for count as wanted
        // when slots are planned, and its reads after, so the experts it used
        // are the last to give way. Left out, the lookups evicted experts
        // still in use and read 40 percent more from storage than the exact
        // way on a 105 GB model under a 34 GB limit.
        self.record_expert_demand(
            layer_index,
            layer_plan.expert_capacity,
            &selected_expert_ids,
        );
        let (prefetch_hit_count, prefetch_miss_count) =
            self.consume_prefetch_coverage(layer_index, &route_coverage);
        let mut read_through = TableReadThrough {
            retained_expert_count: route_coverage.retained_expert_ids.len(),
            missing_expert_count,
            prefetch_hit_count,
            prefetch_miss_count,
            ..TableReadThrough::default()
        };
        if missing_expert_count > 0 {
            let fetch_started_at = std::time::Instant::now();
            let Some(table_fetch_report) = self.fetch_missing_experts_into_table(
                runtime,
                layer_index,
                layer_plan,
                &route_coverage.missing_expert_ids,
                &route_coverage.retained_expert_ids,
                warm_slot_count.min(slot_count_now),
                slot_filler,
            )?
            else {
                return Ok(Err(TableRefusal::CannotSeatEveryExpert));
            };
            read_through.table_fetch_report = Some(table_fetch_report);
            read_through.fetch_elapsed = fetch_started_at.elapsed();
            self.record_disk_load(missing_expert_count, 1);
        }
        if !self.write_table_slots(layer_index, routed_expert_assignments, table_slots) {
            return Ok(Err(TableRefusal::CannotSeatEveryExpert));
        }
        self.record_routed_reads(layer_index, &routed_expert_ids);
        Ok(Ok(read_through))
    }

    /// Grows the layer's table now, on the model's own thread, when it has
    /// fewer than `free_slot_floor` free slots and a fetch that found it so
    /// would grow it. Returns whether it grew.
    ///
    /// A lookup inside an evaluation refuses a route whose fetch would grow
    /// the table, and the token is then served twice. Growing between
    /// tokens, by the rule a fetch grows by, means a lookup finds either
    /// room for a whole route or a table that is as large as it may get, in
    /// which case an expert gives way. Measured before this: 21 of 339
    /// reply tokens refused on a 35B-A3B at a 16 GB limit, every one for
    /// this reason.
    ///
    /// The caller must have evaluated every graph that reads this layer's
    /// table: a table that grows is replaced.
    ///
    /// # Errors
    /// Fails when the larger table cannot be built.
    pub fn grow_table_ahead_of_lookups(
        &mut self,
        runtime: &MlxRuntime,
        layer_index: usize,
        warm_slot_count: usize,
        free_slot_floor: usize,
    ) -> Result<bool, MlxRuntimeError> {
        let resident_payload_bytes = self.resident_payload_bytes;
        let astro_expert_cache_ceiling_bytes = self.capacity_bytes;
        let Some(Some(table)) = self.tables_by_layer.get_mut(layer_index) else {
            return Ok(false);
        };
        let free_slot_count = table.slot_count() - table.occupied_slot_count;
        if table.occupied_slot_count == 0 || free_slot_count >= free_slot_floor {
            return Ok(false);
        }
        let Some(grown_resident_payload_bytes) = grow_table_within_budget(
            runtime,
            table,
            resident_payload_bytes,
            astro_expert_cache_ceiling_bytes,
            warm_slot_count,
            free_slot_floor - free_slot_count,
        )?
        else {
            return Ok(false);
        };
        self.resident_payload_bytes = grown_resident_payload_bytes;
        Ok(true)
    }

    /// Whether a fetch of `new_expert_count` experts into the layer's table
    /// would first replace the table with a larger one.
    fn would_grow_table_for(
        &self,
        layer_index: usize,
        new_expert_count: usize,
        warm_slot_count: usize,
    ) -> bool {
        let Some(Some(table)) = self.tables_by_layer.get(layer_index) else {
            return false;
        };
        let free_slot_count = table.slot_count() - table.occupied_slot_count;
        let free_slot_deficit = new_expert_count.saturating_sub(free_slot_count);
        free_slot_deficit > 0
            && planned_table_growth(
                table,
                self.resident_payload_bytes,
                self.capacity_bytes,
                warm_slot_count,
                free_slot_deficit,
            )
            .is_some()
    }

    /// Writes each assignment's slot; `false`, with some slots written, when
    /// the table lacks one of the experts.
    fn write_table_slots(
        &self,
        layer_index: usize,
        routed_expert_assignments: &[[u8; 4]],
        table_slots: &mut [[u8; 4]],
    ) -> bool {
        let Some(Some(table)) = self.tables_by_layer.get(layer_index) else {
            return false;
        };
        for (routed_expert_assignment, table_slot) in
            routed_expert_assignments.iter().zip(table_slots)
        {
            let expert_id = u32::from_ne_bytes(*routed_expert_assignment) as usize;
            let Some(slot) = table
                .slot_by_expert_id
                .get(&expert_id)
                .and_then(|slot| u32::try_from(*slot).ok())
            else {
                return false;
            };
            *table_slot = slot.to_ne_bytes();
        }
        true
    }
}
