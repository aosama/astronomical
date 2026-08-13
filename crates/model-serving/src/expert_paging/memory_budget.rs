//! Fail-closed, configured MLX memory budget enforcement for expert paging.
//!
//! This module observes process-local MLX counters against the configured
//! worker ceiling. It never changes wired limits, sysctl state, allocator
//! limits, or memory capabilities. The projection accounts for exact pending
//! page bytes in addition to declared dense/page payload bytes.
//!
//! The projection uses Astronomical's existing `MlxRuntime::memory_snapshot()`
//! and preserves fail-closed admission when a pending page would exceed the
//! configured worker ceiling.

use std::cell::Cell;

use thiserror::Error;

use astronomical_runtime_integration::MlxRuntime;

/// Typed failures during memory budget enforcement.
#[derive(Debug, Error)]
pub enum MemoryBudgetError {
    #[error(
        "MLX memory budget exceeded at stage {stage:?}: projected={projected_bytes}, active={active_bytes}, allocator_cache={allocator_cache_bytes}, configured_cap={configured_cap_bytes}"
    )]
    BudgetExceeded {
        stage: String,
        projected_bytes: u64,
        active_bytes: u64,
        allocator_cache_bytes: u64,
        configured_cap_bytes: u64,
    },
    #[error("failed to read GPU memory counters: {0}")]
    MlxRuntime(#[from] astronomical_runtime_integration::MlxRuntimeError),
}

/// Observed counters and conservative projected usage for one runtime stage.
///
/// The `observed_transient_high_water_bytes` field reserves space for
/// computation buffers that the forward pass allocates between expert-page
/// loads. Without this reservation, the retention ceiling would allow too many
/// expert pages to stay resident, leaving insufficient room for transient
/// computation and causing the next forward to exceed the MLX ceiling.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MemoryBudgetSnapshot {
    pub stage: String,
    pub active_bytes: u64,
    pub allocator_cache_bytes: u64,
    pub pending_allocation_bytes: u64,
    pub projected_bytes: u64,
    pub configured_cap_bytes: u64,
    pub maximum_expert_page_bytes: u64,
    /// Peak transient MLX memory observed across completed forward passes with
    /// a similar execution shape. This is the maximum amount of active memory
    /// that temporary computation buffers (attention KV growth, intermediate
    /// tensors, etc.) are expected to need on top of the stable baseline.
    /// Setting this to zero disables the reservation.
    pub observed_transient_high_water_bytes: u64,
}

impl MemoryBudgetSnapshot {
    /// Reuses the observed counters with a candidate live ceiling.
    #[must_use]
    pub fn with_configured_cap_bytes(&self, configured_cap_bytes: u64) -> Self {
        Self {
            configured_cap_bytes,
            ..self.clone()
        }
    }

    /// Returns whether the projected request fits the configured worker cap.
    ///
    /// MLX's patched Metal allocator enforces `active_memory + allocation <= limit`.
    /// The `active_memory_` counter increases by the allocation size regardless of
    /// whether the buffer was reused from cache or freshly allocated. The allocator
    /// cache (`cache_memory_`) is a separate counter that does not participate in
    /// the enforcement check. Therefore the correct projection is simply
    /// `active + pending`, not `active + cache + pending`.
    #[must_use]
    pub fn within_cap(&self) -> bool {
        projected_active_memory_after_allocation(self.active_bytes, self.pending_allocation_bytes)
            .is_some_and(|projected_bytes| projected_bytes <= self.configured_cap_bytes)
    }

    /// Returns whether releasing unused MLX allocator buffers before the
    /// allocation would prevent total GPU memory from exceeding the ceiling.
    ///
    /// The enforcement check (`active + pending <= cap`) can pass while total
    /// GPU memory (`active + cache + pending`) still exceeds the cap. In that
    /// case, proactively clearing the cache avoids triggering MLX's internal
    /// `gc_limit_` release at allocation time, which adds latency.
    #[must_use]
    pub fn should_clear_allocator_cache_to_reduce_total_gpu_memory(&self) -> bool {
        if self.allocator_cache_bytes == 0 || !self.within_cap() {
            return false;
        }
        let Some(total_gpu_memory_after_allocation) = self
            .active_bytes
            .checked_add(self.allocator_cache_bytes)
            .and_then(|active_plus_cache| {
                active_plus_cache.checked_add(self.pending_allocation_bytes)
            })
        else {
            return true;
        };
        total_gpu_memory_after_allocation > self.configured_cap_bytes
    }
}

/// Projects the MLX `active_memory_` counter after a single allocation.
///
/// MLX's patched Metal allocator increments `active_memory_` by the buffer size
/// on every allocation, whether the buffer was reused from cache or freshly
/// created. The enforcement check is `active + allocation > allowed`. The
/// allocator cache is a separate counter that does not affect this projection.
fn projected_active_memory_after_allocation(
    active_bytes: u64,
    pending_allocation_bytes: u64,
) -> Option<u64> {
    active_bytes.checked_add(pending_allocation_bytes)
}

/// Bounds one routed page before lazy router output becomes host-visible.
///
/// A multi-token route can select at most one distinct expert per assignment,
/// while repeated assignments and the layer's finite capacity can only reduce
/// that count. Returning `None` preserves fail-closed behavior on arithmetic
/// overflow.
#[must_use]
pub fn maximum_possible_expert_route_payload_bytes(
    payload_bytes_per_expert: u64,
    expert_capacity: usize,
    selected_expert_assignment_count: usize,
) -> Option<u64> {
    let maximum_distinct_expert_count = expert_capacity.min(selected_expert_assignment_count);
    let maximum_distinct_expert_count = u64::try_from(maximum_distinct_expert_count).ok()?;
    payload_bytes_per_expert.checked_mul(maximum_distinct_expert_count)
}

/// Computes the expert-weight retention ceiling from current MLX residency.
///
/// In plain terms: keep non-expert memory, make room for the exact expert pages
/// this route is missing, reserve room for one ordinary future route, AND
/// reserve room for the transient computation buffers that the forward pass
/// allocates between expert-page loads. Existing retained pages are the only
/// elastic category. If everything fits, retention can grow. If it does not,
/// shrink old retention by only the deficit. Incoming pages are counted once as
/// future retained payload, never once as an allocation and again as retained
/// memory.
///
/// The `observed_transient_high_water_bytes` reservation ensures that after
/// retaining expert pages, enough headroom remains for attention KV growth,
/// intermediate tensors, and other transient allocations that peak during
/// the forward pass and are released afterward.
#[must_use]
pub fn automatic_expert_weight_memory_cache_maximum_size_bytes(
    memory_budget_snapshot: &MemoryBudgetSnapshot,
    current_expert_weight_memory_cache_payload_bytes: u64,
    pending_retained_expert_payload_bytes: u64,
) -> u64 {
    let future_expert_page_reserve_bytes = memory_budget_snapshot.maximum_expert_page_bytes;
    let pending_or_future_expert_page_reserve_bytes = if pending_retained_expert_payload_bytes == 0
    {
        memory_budget_snapshot
            .pending_allocation_bytes
            .max(future_expert_page_reserve_bytes)
    } else {
        let Some(pending_and_future_expert_page_reserve_bytes) = memory_budget_snapshot
            .pending_allocation_bytes
            .checked_add(future_expert_page_reserve_bytes)
        else {
            return 0;
        };
        pending_and_future_expert_page_reserve_bytes
    };
    let Some(post_load_retained_expert_payload_bytes) =
        current_expert_weight_memory_cache_payload_bytes
            .checked_add(pending_retained_expert_payload_bytes)
    else {
        return 0;
    };
    let live_reserved_bytes = projected_active_memory_after_allocation(
        memory_budget_snapshot.active_bytes,
        pending_or_future_expert_page_reserve_bytes,
    )
    .unwrap_or(u64::MAX);
    // Reserve space for the transient computation buffers (attention KV growth,
    // intermediate tensors) that peak during the forward pass. Without this
    // reservation, the retention ceiling would allow too many expert pages to
    // stay resident, leaving insufficient headroom for forward-pass computation
    // and causing the next request to exceed the MLX active-memory ceiling.
    let effective_cap_bytes = memory_budget_snapshot
        .configured_cap_bytes
        .saturating_sub(memory_budget_snapshot.observed_transient_high_water_bytes);
    // `active_bytes` already includes retained expert arrays. Add the current
    // retained payload back after subtracting total live residency. If the
    // pending page will become retained after this load, include it once in the
    // post-load target while retaining a distinct future routed-page reserve.
    let automatic_maximum_size_bytes = if live_reserved_bytes <= effective_cap_bytes {
        post_load_retained_expert_payload_bytes
            .saturating_add(effective_cap_bytes - live_reserved_bytes)
    } else {
        post_load_retained_expert_payload_bytes
            .saturating_sub(live_reserved_bytes - effective_cap_bytes)
    };

    automatic_maximum_size_bytes
}

/// Enforces the worker's resolved MLX memory ceiling at operation boundaries.
#[derive(Debug)]
pub struct LiveMetalBudget {
    maximum_expert_page_bytes: u64,
    configured_cap_bytes: u64,
    /// Peak transient MLX memory observed across completed forward passes.
    /// Interior mutability allows updating from shared references during forward
    /// pass telemetry collection without requiring exclusive access to the model.
    observed_transient_high_water_bytes: Cell<u64>,
}

impl LiveMetalBudget {
    /// Creates a budget using the worker's configured MLX wired-memory ceiling.
    pub fn new(maximum_expert_page_bytes: u64, configured_cap_bytes: u64) -> Self {
        Self {
            maximum_expert_page_bytes,
            configured_cap_bytes,
            observed_transient_high_water_bytes: Cell::new(0),
        }
    }

    /// Replaces the active worker ceiling used by future budget projections.
    pub fn update_configured_cap_bytes(&mut self, configured_cap_bytes: u64) {
        self.configured_cap_bytes = configured_cap_bytes;
    }

    /// Returns the active worker ceiling used by future budget projections.
    #[must_use]
    pub const fn configured_cap_bytes(&self) -> u64 {
        self.configured_cap_bytes
    }

    /// Returns the largest single expert page reserved by this budget.
    #[must_use]
    pub const fn maximum_expert_page_bytes(&self) -> u64 {
        self.maximum_expert_page_bytes
    }

    /// Returns the observed transient high-water mark for forward-pass
    /// computation buffers. Zero before any forward completes.
    #[must_use]
    pub fn observed_transient_high_water_bytes(&self) -> u64 {
        self.observed_transient_high_water_bytes.get()
    }

    /// Updates the observed transient high-water mark from the adaptive RAM
    /// growth guard after a completed forward pass.
    pub fn update_observed_transient_high_water_bytes(
        &self,
        observed_transient_high_water_bytes: u64,
    ) {
        self.observed_transient_high_water_bytes
            .set(observed_transient_high_water_bytes);
    }

    /// Check live counters before/after a bounded operation and return evidence.
    ///
    /// Fails closed when projected local MLX memory exceeds the configured cap.
    pub fn check(
        &self,
        runtime: &MlxRuntime,
        stage: &str,
        pending_allocation_bytes: u64,
    ) -> Result<MemoryBudgetSnapshot, MemoryBudgetError> {
        let mut snapshot = self.snapshot(runtime, stage, pending_allocation_bytes)?;
        if snapshot.should_clear_allocator_cache_to_reduce_total_gpu_memory() {
            tracing::debug!(
                stage,
                allocator_cache_bytes = snapshot.allocator_cache_bytes,
                projected_bytes = snapshot.projected_bytes,
                configured_cap_bytes = snapshot.configured_cap_bytes,
                "synchronizing in-flight GPU work and releasing reclaimable MLX allocator memory before expert-page rejection"
            );
            runtime.synchronize_gpu_stream_and_clear_allocator_cache()?;
            snapshot = self.snapshot(runtime, stage, pending_allocation_bytes)?;
        }
        if !snapshot.within_cap() {
            return Err(MemoryBudgetError::BudgetExceeded {
                stage: stage.to_owned(),
                projected_bytes: snapshot.projected_bytes,
                active_bytes: snapshot.active_bytes,
                allocator_cache_bytes: snapshot.allocator_cache_bytes,
                configured_cap_bytes: snapshot.configured_cap_bytes,
            });
        }
        Ok(snapshot)
    }

    /// Captures the same live projection as [`Self::check`] without rejecting
    /// an over-budget state, allowing retained expert pages to be evicted first.
    pub fn snapshot(
        &self,
        runtime: &MlxRuntime,
        stage: &str,
        pending_allocation_bytes: u64,
    ) -> Result<MemoryBudgetSnapshot, MemoryBudgetError> {
        let snapshot = runtime.memory_snapshot()?;
        let active_bytes = snapshot.active_memory_bytes() as u64;
        let allocator_cache_bytes = snapshot.allocator_cache_memory_bytes() as u64;

        let projected_bytes =
            projected_active_memory_after_allocation(active_bytes, pending_allocation_bytes)
                .ok_or_else(|| MemoryBudgetError::BudgetExceeded {
                    stage: stage.to_owned(),
                    projected_bytes: u64::MAX,
                    active_bytes,
                    allocator_cache_bytes,
                    configured_cap_bytes: self.configured_cap_bytes,
                })?;

        Ok(MemoryBudgetSnapshot {
            stage: stage.to_owned(),
            active_bytes,
            allocator_cache_bytes,
            pending_allocation_bytes,
            projected_bytes,
            configured_cap_bytes: self.configured_cap_bytes,
            maximum_expert_page_bytes: self.maximum_expert_page_bytes,
            observed_transient_high_water_bytes: self.observed_transient_high_water_bytes.get(),
        })
    }
}
