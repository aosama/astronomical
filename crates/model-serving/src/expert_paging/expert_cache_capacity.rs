//! Complete-layer admission and partial expert-page capacity allocation.

use super::expert_cache::ExpertWeightMemoryCache;

impl<ExpertPage> ExpertWeightMemoryCache<ExpertPage>
where
    ExpertPage: super::ExpertWeightPage,
{
    pub(super) fn current_hybrid_retention_payload_byte_count(&self) -> Option<u64> {
        let paged_layer_route_floor_bytes =
            self.paged_layer_decode_route_floor_payload_byte_count_excluding(None);
        if paged_layer_route_floor_bytes > self.maximum_resident_payload_byte_count {
            return None;
        }
        let complete_layer_payload_bytes =
            self.retained_complete_layer_payload_byte_count_excluding(None);
        Some(complete_layer_payload_bytes.saturating_add(paged_layer_route_floor_bytes))
    }

    pub(crate) fn can_retain_selected_expert_payload(
        &self,
        layer_index: usize,
        selected_expert_payload_byte_count: u64,
    ) -> bool {
        selected_expert_payload_byte_count
            <= self.maximum_layer_resident_payload_byte_count(layer_index)
    }

    pub(crate) fn can_physically_retain_complete_layer_expert_payload(
        &self,
        layer_index: usize,
        complete_layer_expert_payload_byte_count: u64,
    ) -> bool {
        self.projected_complete_layer_resident_payload_byte_count(
            layer_index,
            complete_layer_expert_payload_byte_count,
        )
        .is_some_and(|projected_resident_payload_byte_count| {
            projected_resident_payload_byte_count <= self.maximum_resident_payload_byte_count
        })
    }

    /// Returns whether a complete layer fits while preserving affordable decode routes.
    pub fn can_retain_complete_layer_expert_payload(
        &self,
        layer_index: usize,
        complete_layer_expert_payload_byte_count: u64,
    ) -> bool {
        let Some(projected_resident_payload_byte_count) = self
            .projected_complete_layer_resident_payload_byte_count(
                layer_index,
                complete_layer_expert_payload_byte_count,
            )
        else {
            return false;
        };
        if projected_resident_payload_byte_count > self.maximum_resident_payload_byte_count {
            return false;
        }

        let projected_paged_layer_route_floor_bytes =
            self.paged_layer_decode_route_floor_payload_byte_count_excluding(Some(layer_index));
        if projected_paged_layer_route_floor_bytes > self.maximum_resident_payload_byte_count {
            return true;
        }
        let projected_complete_layer_payload_bytes = self
            .retained_complete_layer_payload_byte_count_excluding(Some(layer_index))
            .saturating_add(complete_layer_expert_payload_byte_count);
        projected_complete_layer_payload_bytes
            .saturating_add(projected_paged_layer_route_floor_bytes)
            <= self.maximum_resident_payload_byte_count
    }

    fn projected_complete_layer_resident_payload_byte_count(
        &self,
        layer_index: usize,
        complete_layer_expert_payload_byte_count: u64,
    ) -> Option<u64> {
        let current_layer_resident_payload_byte_count = self
            .resident_payload_byte_count_by_layer
            .get(layer_index)
            .copied()?;
        Some(
            self.resident_payload_byte_count
                .saturating_sub(current_layer_resident_payload_byte_count)
                .saturating_add(complete_layer_expert_payload_byte_count),
        )
    }

    pub(super) fn maximum_layer_resident_payload_byte_count(&self, layer_index: usize) -> u64 {
        if self
            .complete_layer_expert_weights
            .get(layer_index)
            .is_none_or(Option::is_some)
        {
            return 0;
        }
        let paged_layer_count = self
            .complete_layer_expert_weights
            .iter()
            .filter(|complete_layer| complete_layer.is_none())
            .count() as u64;
        if paged_layer_count == 0 {
            return 0;
        }
        let complete_layer_payload_bytes =
            self.retained_complete_layer_payload_byte_count_excluding(None);
        let partial_page_budget_bytes = self
            .maximum_resident_payload_byte_count
            .saturating_sub(complete_layer_payload_bytes);
        let paged_layer_route_floor_bytes =
            self.paged_layer_decode_route_floor_payload_byte_count_excluding(None);
        let paged_layer_ordinal = self.complete_layer_expert_weights[..layer_index]
            .iter()
            .filter(|complete_layer| complete_layer.is_none())
            .count() as u64;
        if paged_layer_route_floor_bytes > partial_page_budget_bytes {
            return equal_share_with_remainder(
                partial_page_budget_bytes,
                paged_layer_count,
                paged_layer_ordinal,
            );
        }

        let selected_layer_route_floor_bytes = self
            .minimum_decode_route_payload_byte_count_by_layer
            .get(layer_index)
            .copied()
            .unwrap_or(0);
        selected_layer_route_floor_bytes.saturating_add(equal_share_with_remainder(
            partial_page_budget_bytes.saturating_sub(paged_layer_route_floor_bytes),
            paged_layer_count,
            paged_layer_ordinal,
        ))
    }

    fn retained_complete_layer_payload_byte_count_excluding(
        &self,
        excluded_layer_index: Option<usize>,
    ) -> u64 {
        self.complete_layer_expert_weights
            .iter()
            .enumerate()
            .filter(|(layer_index, _)| Some(*layer_index) != excluded_layer_index)
            .filter_map(|(_, complete_layer)| complete_layer.as_ref())
            .map(|complete_layer| complete_layer.resident_payload_byte_count)
            .fold(0u64, u64::saturating_add)
    }

    fn paged_layer_decode_route_floor_payload_byte_count_excluding(
        &self,
        excluded_layer_index: Option<usize>,
    ) -> u64 {
        self.minimum_decode_route_payload_byte_count_by_layer
            .iter()
            .enumerate()
            .filter(|(layer_index, _)| Some(*layer_index) != excluded_layer_index)
            .filter(|(layer_index, _)| self.complete_layer_expert_weights[*layer_index].is_none())
            .map(|(_, route_floor_payload_byte_count)| *route_floor_payload_byte_count)
            .fold(0u64, u64::saturating_add)
    }
}

fn equal_share_with_remainder(
    payload_byte_count: u64,
    layer_count: u64,
    layer_ordinal: u64,
) -> u64 {
    let equal_share_bytes = payload_byte_count / layer_count;
    let remainder_bytes = payload_byte_count % layer_count;
    equal_share_bytes + u64::from(layer_ordinal < remainder_bytes)
}
