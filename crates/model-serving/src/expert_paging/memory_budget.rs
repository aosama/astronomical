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
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MemoryBudgetSnapshot {
    pub stage: String,
    pub active_bytes: u64,
    pub allocator_cache_bytes: u64,
    pub pending_allocation_bytes: u64,
    pub projected_bytes: u64,
    pub configured_cap_bytes: u64,
    pub maximum_expert_page_bytes: u64,
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
    #[must_use]
    pub fn within_cap(&self) -> bool {
        self.projected_mlx_memory_bytes()
            .is_some_and(|projected_mlx_memory_bytes| {
                projected_mlx_memory_bytes <= self.configured_cap_bytes
            })
    }

    /// Returns whether releasing unused MLX allocator buffers would make this
    /// otherwise rejected allocation fit without reducing active model memory.
    #[must_use]
    pub fn should_reclaim_allocator_cache_before_rejecting(&self) -> bool {
        if self.allocator_cache_bytes == 0 || self.within_cap() {
            return false;
        }
        self.active_bytes
            .checked_add(self.pending_allocation_bytes)
            .is_some_and(|projected_mlx_memory_bytes_without_allocator_cache| {
                projected_mlx_memory_bytes_without_allocator_cache <= self.configured_cap_bytes
            })
    }

    fn projected_mlx_memory_bytes(&self) -> Option<u64> {
        projected_mlx_memory_bytes_after_allocator_reuse(
            self.active_bytes,
            self.allocator_cache_bytes,
            self.pending_allocation_bytes,
        )
    }
}

fn projected_mlx_memory_bytes_after_allocator_reuse(
    active_bytes: u64,
    allocator_cache_bytes: u64,
    pending_allocation_bytes: u64,
) -> Option<u64> {
    // MLX's pinned Metal allocator first reuses a suitable cached buffer. If no
    // buffer matches, it releases cached buffers before allocating under memory
    // pressure. Only bytes beyond the reclaimable allocator pool can increase
    // system allocation; counting the whole pending page here caused a full
    // cache clear before nearly every expert layer.
    let additional_system_gpu_memory_bytes =
        pending_allocation_bytes.saturating_sub(allocator_cache_bytes);
    active_bytes
        .checked_add(allocator_cache_bytes)?
        .checked_add(additional_system_gpu_memory_bytes)
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
/// The configured cap first reserves local MLX residency and either a temporary
/// current page or both an
/// incoming retained allocation and one future routed page. The incoming
/// retained payload is added once to the post-load target. The process-local
/// total already includes retained expert arrays. Any remaining bytes can grow
/// the retained payload. If those reservations already exceed the cap,
/// the retained payload shrinks by the overage.
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
    let live_reserved_bytes = projected_mlx_memory_bytes_after_allocator_reuse(
        memory_budget_snapshot.active_bytes,
        memory_budget_snapshot.allocator_cache_bytes,
        pending_or_future_expert_page_reserve_bytes,
    )
    .unwrap_or(u64::MAX);
    // `active_bytes` already includes retained expert arrays. Add the current
    // retained payload back after subtracting total live residency. If the
    // pending page will become retained after this load, include it once in the
    // post-load target while retaining a distinct future routed-page reserve.
    let automatic_maximum_size_bytes =
        if live_reserved_bytes <= memory_budget_snapshot.configured_cap_bytes {
            post_load_retained_expert_payload_bytes
                .saturating_add(memory_budget_snapshot.configured_cap_bytes - live_reserved_bytes)
        } else {
            post_load_retained_expert_payload_bytes
                .saturating_sub(live_reserved_bytes - memory_budget_snapshot.configured_cap_bytes)
        };

    automatic_maximum_size_bytes
}

/// Enforces the worker's resolved MLX memory ceiling at operation boundaries.
#[derive(Debug)]
pub struct LiveMetalBudget {
    maximum_expert_page_bytes: u64,
    configured_cap_bytes: u64,
}

impl LiveMetalBudget {
    /// Creates a budget using the worker's configured MLX wired-memory ceiling.
    pub fn new(maximum_expert_page_bytes: u64, configured_cap_bytes: u64) -> Self {
        Self {
            maximum_expert_page_bytes,
            configured_cap_bytes,
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
        if snapshot.should_reclaim_allocator_cache_before_rejecting() {
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

        let projected_bytes = projected_mlx_memory_bytes_after_allocator_reuse(
            active_bytes,
            allocator_cache_bytes,
            pending_allocation_bytes,
        )
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
        })
    }
}
