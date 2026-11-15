//! The expert cache: one capacity across all layers, one eviction order.
//!
//! The cache this replaces divided memory by layer before a reply began: as
//! many whole layers as fit, and a tiny table for each layer left over. A
//! recorded reply showed what that costs. A 48-layer model held 11,568
//! experts and the reply used 10,033, yet three starved layers read about 13
//! experts from storage on every token while 2,008 held experts were never
//! used once. Memory was sufficient; the division was wrong, and nothing
//! could move memory from an idle expert in one layer to a busy expert in
//! another.
//!
//! So this cache has no per-layer division of its own. A miss gives up the
//! least valuable expert anywhere. `LayerSlotLimit::FixedPerLayer` exists to
//! reproduce the old division in a replay and for an engine whose weight
//! tables cannot yet change size; it is not the design.

use std::collections::BTreeSet;

use super::cache_record::ExpertCacheRecord;

/// One expert of one layer.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ExpertKey {
    pub layer_index: usize,
    pub expert_id: u16,
}

/// What the cache needs to know about a layer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExpertCacheLayerGeometry {
    pub expert_count: u16,
    pub expert_payload_bytes: u64,
}

/// Which held expert is given up first.
///
/// Both are textbook orders. On the recorded replies they read exactly the
/// same number of experts, so the cache defaults to the simpler one and the
/// replay test keeps comparing them, in case another kind of text differs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EvictionOrder {
    LeastRecentlyUsed,
    LeastFrequentlyUsed,
}

/// Whether a layer may hold only so many experts regardless of free memory.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LayerSlotLimit {
    /// Any layer may hold any number of its experts; only bytes limit it.
    SharedAcrossLayers,
    /// Each layer holds at most this many experts, and a full layer gives up
    /// one of its own experts for a new one.
    FixedPerLayer(Vec<usize>),
}

/// What serving one layer of one token did.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ServedRoute {
    /// Experts read from storage and now held.
    pub read_expert_ids: Vec<u16>,
    /// Experts read from storage and not kept, because nothing could be
    /// given up for them.
    pub unheld_expert_ids: Vec<u16>,
    /// Experts given up, in any layer, to make room for the reads.
    pub evicted: Vec<ExpertKey>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct HeldExpert {
    last_used_token: u64,
    use_count: u64,
}

/// Lower ranks are given up first.
type EvictionRank = (u64, u64);

enum RoomShortage {
    LayerSlots,
    Bytes,
}

#[derive(Debug)]
pub struct ExpertCache {
    layer_geometries: Vec<ExpertCacheLayerGeometry>,
    capacity_bytes: u64,
    held_bytes: u64,
    eviction_order: EvictionOrder,
    layer_slot_limit: LayerSlotLimit,
    held_by_layer: Vec<Vec<Option<HeldExpert>>>,
    held_count_by_layer: Vec<usize>,
    /// Every held expert ordered by rank, so the next expert to give up is
    /// found without scanning twelve thousand entries on each miss.
    eviction_queue: BTreeSet<(EvictionRank, ExpertKey)>,
    current_token: u64,
    record: ExpertCacheRecord,
}

impl ExpertCache {
    #[must_use]
    pub fn new(
        layer_geometries: Vec<ExpertCacheLayerGeometry>,
        capacity_bytes: u64,
        eviction_order: EvictionOrder,
        layer_slot_limit: LayerSlotLimit,
    ) -> Self {
        let mut held_by_layer = Vec::with_capacity(layer_geometries.len());
        let mut expert_counts = Vec::with_capacity(layer_geometries.len());
        for layer_geometry in &layer_geometries {
            held_by_layer.push(vec![None; usize::from(layer_geometry.expert_count)]);
            expert_counts.push(layer_geometry.expert_count);
        }
        Self {
            held_count_by_layer: vec![0; layer_geometries.len()],
            record: ExpertCacheRecord::for_layers(expert_counts.into_iter()),
            layer_geometries,
            capacity_bytes,
            held_bytes: 0,
            eviction_order,
            layer_slot_limit,
            held_by_layer,
            eviction_queue: BTreeSet::new(),
            current_token: 0,
        }
    }

    #[must_use]
    pub const fn held_bytes(&self) -> u64 {
        self.held_bytes
    }

    #[must_use]
    pub const fn capacity_bytes(&self) -> u64 {
        self.capacity_bytes
    }

    #[must_use]
    pub fn is_held(&self, expert: ExpertKey) -> bool {
        self.held_by_layer[expert.layer_index][usize::from(expert.expert_id)].is_some()
    }

    #[must_use]
    pub fn held_expert_count(&self, layer_index: usize) -> usize {
        self.held_count_by_layer[layer_index]
    }

    #[must_use]
    pub const fn record(&self) -> &ExpertCacheRecord {
        &self.record
    }

    /// Holds an expert that was loaded ahead of any token, such as the
    /// layers read in while a reply is prepared. Gives up nothing: returns
    /// `false` when there is no room, and the expert stays out.
    pub fn seat(&mut self, expert: ExpertKey) -> bool {
        if self.is_held(expert) {
            return true;
        }
        if self.room_shortage_for(expert).is_some() {
            return false;
        }
        self.hold(expert);
        self.record.record_seated(expert);
        true
    }

    /// Starts the next token. Experts a token uses are never given up for
    /// another expert of the same token.
    pub fn begin_token(&mut self) {
        self.current_token += 1;
        self.record.begin_token();
    }

    /// Serves one layer's route for the current token: counts the hits,
    /// reads what is missing and gives up what the reads need room for.
    pub fn serve(&mut self, layer_index: usize, routed_expert_ids: &[u16]) -> ServedRoute {
        let mut served_route = ServedRoute::default();
        let mut missing_experts = Vec::new();
        for &expert_id in routed_expert_ids {
            let expert = ExpertKey {
                layer_index,
                expert_id,
            };
            let is_hit = self.is_held(expert);
            self.record.record_lookup(expert, is_hit);
            if is_hit {
                self.mark_used(expert);
            } else {
                missing_experts.push(expert);
            }
        }
        for expert in missing_experts {
            self.read_missing_expert(expert, &mut served_route);
        }
        served_route
    }

    fn read_missing_expert(&mut self, expert: ExpertKey, served_route: &mut ServedRoute) {
        // A route may name the same expert twice; the first read holds it.
        if self.is_held(expert) {
            return;
        }
        let is_kept = self.make_room_for(expert, &mut served_route.evicted);
        self.record.record_read(expert, is_kept);
        if is_kept {
            self.hold(expert);
            self.mark_used(expert);
            served_route.read_expert_ids.push(expert.expert_id);
        } else {
            served_route.unheld_expert_ids.push(expert.expert_id);
        }
    }

    /// Changes how many bytes the cache may hold and gives up, in eviction
    /// order, whatever no longer fits.
    pub fn set_capacity(&mut self, capacity_bytes: u64) -> Vec<ExpertKey> {
        self.capacity_bytes = capacity_bytes;
        let mut evicted = Vec::new();
        while self.held_bytes > self.capacity_bytes {
            let Some(victim) = self.least_valuable_expert_anywhere(false) else {
                break;
            };
            self.evict(None, victim);
            evicted.push(victim);
        }
        evicted
    }

    fn make_room_for(&mut self, needed: ExpertKey, evicted: &mut Vec<ExpertKey>) -> bool {
        while let Some(room_shortage) = self.room_shortage_for(needed) {
            let victim = match room_shortage {
                RoomShortage::LayerSlots => self.least_valuable_expert_in(needed.layer_index),
                RoomShortage::Bytes => self.least_valuable_expert_anywhere(true),
            };
            let Some(victim) = victim else {
                return false;
            };
            self.evict(Some(needed), victim);
            evicted.push(victim);
        }
        true
    }

    fn room_shortage_for(&self, needed: ExpertKey) -> Option<RoomShortage> {
        if let LayerSlotLimit::FixedPerLayer(slot_counts) = &self.layer_slot_limit {
            let slot_count = slot_counts.get(needed.layer_index).copied().unwrap_or(0);
            if self.held_count_by_layer[needed.layer_index] >= slot_count {
                return Some(RoomShortage::LayerSlots);
            }
        }
        let needed_bytes = self.layer_geometries[needed.layer_index].expert_payload_bytes;
        if self.held_bytes.saturating_add(needed_bytes) > self.capacity_bytes {
            return Some(RoomShortage::Bytes);
        }
        None
    }

    fn least_valuable_expert_anywhere(&self, spare_current_token: bool) -> Option<ExpertKey> {
        for (_eviction_rank, expert) in &self.eviction_queue {
            if !(spare_current_token && self.was_used_by_current_token(*expert)) {
                return Some(*expert);
            }
        }
        None
    }

    fn least_valuable_expert_in(&self, layer_index: usize) -> Option<ExpertKey> {
        let mut least_valuable: Option<(EvictionRank, ExpertKey)> = None;
        for (expert_id, held_expert) in self.held_by_layer[layer_index].iter().enumerate() {
            let Some(held_expert) = held_expert else {
                continue;
            };
            if held_expert.last_used_token == self.current_token {
                continue;
            }
            let candidate = (
                self.eviction_rank(*held_expert),
                ExpertKey {
                    layer_index,
                    expert_id: u16::try_from(expert_id).unwrap_or(u16::MAX),
                },
            );
            if least_valuable.is_none_or(|current_least| candidate < current_least) {
                least_valuable = Some(candidate);
            }
        }
        least_valuable.map(|(_eviction_rank, expert)| expert)
    }

    fn was_used_by_current_token(&self, expert: ExpertKey) -> bool {
        self.held_by_layer[expert.layer_index][usize::from(expert.expert_id)]
            .is_some_and(|held_expert| held_expert.last_used_token == self.current_token)
    }

    const fn eviction_rank(&self, held_expert: HeldExpert) -> EvictionRank {
        match self.eviction_order {
            EvictionOrder::LeastRecentlyUsed => (held_expert.last_used_token, 0),
            EvictionOrder::LeastFrequentlyUsed => {
                (held_expert.use_count, held_expert.last_used_token)
            }
        }
    }

    fn hold(&mut self, expert: ExpertKey) {
        let held_expert = HeldExpert {
            last_used_token: 0,
            use_count: 0,
        };
        self.held_by_layer[expert.layer_index][usize::from(expert.expert_id)] = Some(held_expert);
        self.held_count_by_layer[expert.layer_index] += 1;
        self.held_bytes += self.layer_geometries[expert.layer_index].expert_payload_bytes;
        self.eviction_queue
            .insert((self.eviction_rank(held_expert), expert));
    }

    fn mark_used(&mut self, expert: ExpertKey) {
        let Some(held_expert) =
            self.held_by_layer[expert.layer_index][usize::from(expert.expert_id)]
        else {
            return;
        };
        self.eviction_queue
            .remove(&(self.eviction_rank(held_expert), expert));
        let used_expert = HeldExpert {
            last_used_token: self.current_token,
            use_count: held_expert.use_count + 1,
        };
        self.held_by_layer[expert.layer_index][usize::from(expert.expert_id)] = Some(used_expert);
        self.eviction_queue
            .insert((self.eviction_rank(used_expert), expert));
    }

    fn evict(&mut self, needed: Option<ExpertKey>, victim: ExpertKey) {
        let Some(held_expert) =
            self.held_by_layer[victim.layer_index][usize::from(victim.expert_id)].take()
        else {
            return;
        };
        self.eviction_queue
            .remove(&(self.eviction_rank(held_expert), victim));
        self.held_count_by_layer[victim.layer_index] -= 1;
        self.held_bytes -= self.layer_geometries[victim.layer_index].expert_payload_bytes;
        self.record.record_eviction(needed, victim);
    }
}
