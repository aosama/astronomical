//! A reply's routes, and what the `astro_expert_cache` held when it began,
//! taken out of the engine so the reply can be replayed without a GPU.
//!
//! Routing depends only on the model and the text, never on memory, so one
//! recorded reply can be replayed against any capacity and any keeping rule
//! (`memory_policy::astro_expert_cache_model`). Replaying the
//! engine's own rule must reproduce the misses the engine counted, which needs
//! the starting point too: the experts each layer held when the reply's first
//! token ran, and the capacity the `astro_expert_cache` had.
//!
//! Recording runs only with performance attribution on, the same switch that
//! makes the engine copy each decode token's route to the host. A reply served
//! with attribution off takes no snapshot and copies nothing.

use crate::qwen3_5_text::model::Qwen3_5Model;

/// What the `astro_expert_cache` looked like when a reply's first token ran.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ReplyStartSnapshot {
    held_expert_ids_by_layer: Vec<Vec<u16>>,
    prompt_demand_by_layer: Vec<Vec<u64>>,
    reply_demand_by_layer: Vec<Vec<u64>>,
    kept_expert_capacity_bytes: u64,
    /// The observation ring's lifetime count when the reply began: every
    /// observation stored after it belongs to this reply.
    stored_observation_count_at_reply_start: u64,
}

impl ReplyStartSnapshot {
    /// A snapshot holding nothing, at `stored_observation_count_at_reply_start`.
    #[cfg(test)]
    pub(crate) const fn empty_at(stored_observation_count_at_reply_start: u64) -> Self {
        Self {
            held_expert_ids_by_layer: Vec::new(),
            prompt_demand_by_layer: Vec::new(),
            reply_demand_by_layer: Vec::new(),
            kept_expert_capacity_bytes: 0,
            stored_observation_count_at_reply_start,
        }
    }

    pub const fn stored_observation_count_at_reply_start(&self) -> u64 {
        self.stored_observation_count_at_reply_start
    }
}

/// One recorded reply, ready to replay.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RoutedReplyRecording {
    /// Experts in one sparse layer.
    pub experts_per_layer: usize,
    /// For each reply token after the first, for each layer, the experts it
    /// routed to. The first reply token comes out of the prompt's last step,
    /// so a reply of `n` tokens records `n - 1` routes. The observation ring
    /// keeps the newest 2,048 decode tokens, so a longer reply records only
    /// its last 2,048.
    pub token_routes: Vec<Vec<Vec<u16>>>,
    /// For each layer, the experts the `astro_expert_cache` held when the
    /// reply's first token ran.
    pub held_expert_ids_by_layer_at_reply_start: Vec<Vec<u16>>,
    /// For each layer, the experts it held when the recording was taken,
    /// after the reply: a layer the reply's plan kept nothing in holds none.
    pub held_expert_ids_by_layer_at_reply_end: Vec<Vec<u16>>,
    /// How often this request's prompt routed to each expert of each layer,
    /// its final chunk counted heavier, when the reply began.
    pub prompt_demand_by_layer_at_reply_start: Vec<Vec<u64>>,
    /// How often earlier replies routed to each expert of each layer, when
    /// this reply began.
    pub reply_demand_by_layer_at_reply_start: Vec<Vec<u64>>,
    /// The `astro_expert_cache` capacity the reply was served under.
    pub kept_expert_capacity_bytes: u64,
    /// Slots each layer not held whole may keep, when the recording was
    /// taken (the engine's warm slot count).
    pub warm_slot_count: usize,
    /// Each capacity the engine applied while the reply was served.
    pub capacity_moments: Vec<memory_policy::astro_expert_cache_model::CapacityMoment>,
    /// One expert's payload bytes in each layer.
    pub expert_payload_bytes_by_layer: Vec<u64>,
}

impl Qwen3_5Model {
    /// Takes the reply-start snapshot when this reply is being recorded.
    pub fn note_reply_start_for_route_recording(&self, is_recording: bool) {
        let (Some(astro_expert_cache), Some(expert_pager), true) = (
            self.astro_expert_cache.as_ref(),
            self.expert_pager.as_ref(),
            is_recording,
        ) else {
            return;
        };
        let held_expert_ids_by_layer = held_expert_ids_by_layer(astro_expert_cache, expert_pager);
        let (prompt_demand_by_layer, reply_demand_by_layer) =
            astro_expert_cache.borrow().demand_counts_snapshot();
        let mut route_observation = self.route_observation.borrow_mut();
        let stored_observation_count_at_reply_start = route_observation.stored_observation_count();
        route_observation.set_reply_start_snapshot(ReplyStartSnapshot {
            held_expert_ids_by_layer,
            prompt_demand_by_layer,
            reply_demand_by_layer,
            kept_expert_capacity_bytes: self.kept_expert_capacity_bytes(),
            stored_observation_count_at_reply_start,
        });
    }

    /// The last recorded reply, or `None` when no reply was recorded since
    /// the model loaded. Taking it clears the snapshot.
    pub fn take_routed_reply_recording(&self) -> Option<RoutedReplyRecording> {
        let expert_pager = self.expert_pager.as_ref()?;
        let mut route_observation = self.route_observation.borrow_mut();
        let reply_start = route_observation.take_reply_start_snapshot()?;
        let capacity_moments = route_observation.take_capacity_moments();
        let reply_token_count = route_observation
            .stored_observation_count()
            .saturating_sub(reply_start.stored_observation_count_at_reply_start);
        let token_routes = route_observation.newest_token_routes(
            usize::try_from(reply_token_count).unwrap_or(usize::MAX),
            expert_pager.layer_plans().len(),
        );
        // The ring keeps only the newest routes; moments are indexed from the
        // reply's start, so they move back by the tokens no longer kept.
        let dropped_token_count = u32::try_from(
            reply_token_count.saturating_sub(u64::try_from(token_routes.len()).unwrap_or(u64::MAX)),
        )
        .unwrap_or(u32::MAX);
        let capacity_moments = memory_policy::astro_expert_cache_model::rebased_capacity_moments(
            &capacity_moments,
            dropped_token_count,
        );
        let mut expert_payload_bytes_by_layer = Vec::new();
        for layer_plan in expert_pager.layer_plans() {
            expert_payload_bytes_by_layer.push(
                crate::qwen3_5_text::expert_streaming::retained_expert_layer_page_payload_bytes(
                    layer_plan, 1,
                )
                .unwrap_or(0),
            );
        }
        Some(RoutedReplyRecording {
            experts_per_layer: expert_pager
                .layer_plans()
                .first()
                .map_or(0, |layer_plan| layer_plan.expert_capacity),
            token_routes,
            held_expert_ids_by_layer_at_reply_start: reply_start.held_expert_ids_by_layer,
            prompt_demand_by_layer_at_reply_start: reply_start.prompt_demand_by_layer,
            reply_demand_by_layer_at_reply_start: reply_start.reply_demand_by_layer,
            held_expert_ids_by_layer_at_reply_end: match self.astro_expert_cache.as_ref() {
                Some(astro_expert_cache) => {
                    held_expert_ids_by_layer(astro_expert_cache, expert_pager)
                }
                None => Vec::new(),
            },
            kept_expert_capacity_bytes: reply_start.kept_expert_capacity_bytes,
            warm_slot_count: self.hot_expert_warm_slot_count(),
            capacity_moments,
            expert_payload_bytes_by_layer,
        })
    }
}

/// What each layer holds now.
fn held_expert_ids_by_layer(
    astro_expert_cache: &std::cell::RefCell<crate::qwen3_5_text::AstroExpertCache>,
    expert_pager: &crate::qwen3_5_text::expert_streaming::expert_pager::Qwen3_5ExpertPager,
) -> Vec<Vec<u16>> {
    let expert_capacity = expert_pager
        .layer_plans()
        .first()
        .map_or(0, |layer_plan| layer_plan.expert_capacity);
    let mut held_expert_ids_by_layer = vec![Vec::new(); expert_pager.layer_plans().len()];
    for layer_residency in astro_expert_cache
        .borrow()
        .topology_snapshot(expert_capacity)
    {
        if let Some(held_expert_ids) = held_expert_ids_by_layer.get_mut(layer_residency.layer_index)
        {
            *held_expert_ids = expert_ids_as_u16(&layer_residency.retained_expert_ids);
        }
    }
    held_expert_ids_by_layer
}

fn expert_ids_as_u16(expert_ids: &[usize]) -> Vec<u16> {
    let mut held_expert_ids = Vec::with_capacity(expert_ids.len());
    for &expert_id in expert_ids {
        held_expert_ids.push(u16::try_from(expert_id).unwrap_or(u16::MAX));
    }
    held_expert_ids
}
