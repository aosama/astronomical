//! Request-scoped retention boundaries.

use super::ExpertWeightMemoryCache;

impl<ExpertPage> ExpertWeightMemoryCache<ExpertPage>
where
    ExpertPage: super::ExpertWeightPage,
{
    /// Applies a request-scoped ceiling without discarding retained experts that
    /// still fit beside the request's dynamic memory.
    pub fn limit_retention_for_request_memory_pressure(
        &mut self,
        maximum_resident_payload_byte_count: u64,
    ) {
        self.request_memory_pressure_maximum_resident_payload_byte_count = Some(
            self.request_memory_pressure_maximum_resident_payload_byte_count
                .map_or(maximum_resident_payload_byte_count, |current_maximum| {
                    current_maximum.min(maximum_resident_payload_byte_count)
                }),
        );
        self.update_maximum_resident_payload_byte_count(maximum_resident_payload_byte_count);
    }

    /// Freezes optional growth until request finalization restores live admission.
    pub fn freeze_retention_growth_for_request_memory_pressure(&mut self) -> bool {
        self.limit_retention_for_request_memory_pressure(self.resident_payload_byte_count);
        true
    }

    /// Restores automatic retention after a pressured request ends.
    pub fn resume_retention_after_request_memory_pressure(&mut self) -> bool {
        if self
            .request_memory_pressure_maximum_resident_payload_byte_count
            .is_none()
        {
            return false;
        }
        self.request_memory_pressure_maximum_resident_payload_byte_count = None;
        self.update_maximum_resident_payload_byte_count(u64::MAX);
        !self.has_complete_expert_layers_for_every_decoder_layer()
    }

    pub(super) fn maximum_resident_payload_byte_count_under_pressure_limits(
        &self,
        live_maximum_resident_payload_byte_count: u64,
    ) -> u64 {
        live_maximum_resident_payload_byte_count.min(
            self.request_memory_pressure_maximum_resident_payload_byte_count
                .unwrap_or(u64::MAX),
        )
    }
}
