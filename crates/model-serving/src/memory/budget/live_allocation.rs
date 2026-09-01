//! Runtime-backed MLX allocation observation and admission.
//!
//! The owner samples MLX and returns policy outcomes. It never clears allocator
//! storage or performs the allocation; those mechanisms remain with callers.

use std::cell::Cell;

use astronomical_runtime_integration::MlxRuntime;
use thiserror::Error;

use crate::memory::{AllocationAdmissionDecision, AllocationAdmissionObservation, MemoryBoundary};

/// Typed failure while sampling or rejecting an MLX allocation.
#[derive(Debug, Error)]
pub enum MlxAllocationAdmissionError {
    #[error(
        "MLX allocation rejected at {stage}: boundary={boundary:?}, shortfall={shortfall_bytes}, active={active_memory_bytes}, pending={pending_allocation_bytes}, ceiling={active_memory_ceiling_bytes}"
    )]
    Rejected {
        stage: String,
        boundary: MemoryBoundary,
        shortfall_bytes: u64,
        active_memory_bytes: u64,
        pending_allocation_bytes: u64,
        active_memory_ceiling_bytes: u64,
    },
    #[error("failed to read MLX memory counters: {0}")]
    MlxRuntime(#[from] astronomical_runtime_integration::MlxRuntimeError),
}

/// Runtime-backed policy state shared by expert allocation boundaries.
#[derive(Debug)]
pub struct MlxAllocationAdmission {
    maximum_expert_page_bytes: u64,
    active_memory_ceiling_bytes: u64,
    observed_transient_high_water_bytes: Cell<u64>,
}

impl MlxAllocationAdmission {
    #[must_use]
    pub fn new(maximum_expert_page_bytes: u64, active_memory_ceiling_bytes: u64) -> Self {
        Self {
            maximum_expert_page_bytes,
            active_memory_ceiling_bytes,
            observed_transient_high_water_bytes: Cell::new(0),
        }
    }

    pub fn update_active_memory_ceiling_bytes(&mut self, active_memory_ceiling_bytes: u64) {
        self.active_memory_ceiling_bytes = active_memory_ceiling_bytes;
    }

    #[must_use]
    pub const fn active_memory_ceiling_bytes(&self) -> u64 {
        self.active_memory_ceiling_bytes
    }

    #[must_use]
    pub const fn maximum_expert_page_bytes(&self) -> u64 {
        self.maximum_expert_page_bytes
    }

    #[must_use]
    pub fn observed_transient_high_water_bytes(&self) -> u64 {
        self.observed_transient_high_water_bytes.get()
    }

    pub fn update_observed_transient_high_water_bytes(
        &self,
        observed_transient_high_water_bytes: u64,
    ) {
        self.observed_transient_high_water_bytes
            .set(observed_transient_high_water_bytes);
    }

    /// Samples one allocation boundary without executing the returned advice.
    pub fn observe(
        &self,
        runtime: &MlxRuntime,
        pending_allocation_bytes: u64,
    ) -> Result<AllocationAdmissionObservation, MlxAllocationAdmissionError> {
        let memory_snapshot = runtime.memory_snapshot()?;
        Ok(AllocationAdmissionObservation::new(
            memory_snapshot.active_memory_bytes() as u64,
            memory_snapshot.allocator_cache_memory_bytes() as u64,
            pending_allocation_bytes,
            self.active_memory_ceiling_bytes,
        ))
    }

    /// Re-samples after optional caller cleanup and rejects any remaining deficit.
    pub fn require_admission(
        &self,
        runtime: &MlxRuntime,
        stage: &str,
        pending_allocation_bytes: u64,
    ) -> Result<AllocationAdmissionDecision, MlxAllocationAdmissionError> {
        let observation = self.observe(runtime, pending_allocation_bytes)?;
        match observation.decide() {
            AllocationAdmissionDecision::Reject {
                boundary,
                shortfall_bytes,
            } => Err(MlxAllocationAdmissionError::Rejected {
                stage: stage.to_owned(),
                boundary,
                shortfall_bytes,
                active_memory_bytes: observation.active_memory_bytes,
                pending_allocation_bytes,
                active_memory_ceiling_bytes: observation.active_memory_ceiling_bytes,
            }),
            decision => Ok(decision),
        }
    }
}
