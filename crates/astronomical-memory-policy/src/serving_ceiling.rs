//! Resolution of the effective MLX memory ceiling from the user's
//! configuration and the machine bound.
//!
//! One source of truth governs the maximum: the machine's physical memory.
//! Astronomical keeps no reserve of its own for the operating system and
//! refuses nothing because a Mac is small (#1902): MLX's own allocation
//! limits and macOS (compression, then swap) handle memory pressure, so an
//! 8 GB Mac may serve slowly but is never turned away. The user's configured
//! value may cap below the bound; with no configuration the machine bound is
//! the ceiling. No graphics-memory recommendation from the kernel or the
//! device participates in the bound; the wired-residency cap below is applied
//! separately, because that one is Metal's own limit.

/// Resolves the machine MLX memory ceiling from the physical memory size.
///
/// The ceiling is the whole of physical memory. A fixed operating-system
/// reserve used to be subtracted here, which left an 8 GB Mac a 1 GB ceiling
/// and so refused every catalog model on the machines the project most wants
/// to reach; that reserve is gone on purpose.
#[must_use]
pub const fn machine_ceiling_from_physical_ram_bytes(physical_ram_bytes: u64) -> u64 {
    physical_ram_bytes
}

/**
 * Resolves the effective ceiling under the configured and machine limits.
 */
#[must_use]
pub const fn effective_mlx_memory_ceiling_bytes(
    configured_mlx_memory_ceiling_bytes: Option<u64>,
    machine_mlx_memory_ceiling_bytes: u64,
) -> u64 {
    match configured_mlx_memory_ceiling_bytes {
        Some(configured_mlx_memory_ceiling_bytes)
            if configured_mlx_memory_ceiling_bytes < machine_mlx_memory_ceiling_bytes =>
        {
            configured_mlx_memory_ceiling_bytes
        }
        _ => machine_mlx_memory_ceiling_bytes,
    }
}

/// Resolves the effective ceiling the engine may plan against once the
/// worker's wired-residency budget is known.
///
/// The wired budget is the only hard GPU-side bound, and it can sit below
/// the machine ceiling: MLX refuses wired claims above Metal's
/// recommended working set, and on a 48 GB machine that recommendation
/// (40,000,028,672 bytes) lands ~2.9 GB under the ceiling
/// (42,949,672,960 bytes). Serving with the higher ceiling while the wired
/// budget stayed lower admitted a 130,538-token prefill whose working set
/// could not be made GPU-resident: the driver failed the command buffer
/// with `kIOGPUCommandBufferCallbackErrorOutOfMemory` and the worker exited
/// (measured 2026-10-16 on stable 0.2.73, issue #1297). Whenever a wired
/// budget exists the engine must plan against the smaller of the two
/// bounds, because the residency planner keeps its activation headroom
/// inside the ceiling it is handed — a truthful ceiling keeps that headroom
/// real. `None` means serving without a wired pool (claim disabled or
/// failed), where pages stay swappable and the ceiling stands unchanged.
///
/// The bound the engine receives is the plannable wired residency, not the
/// raw claim; see [`plannable_wired_residency_bytes`].
#[must_use]
pub const fn effective_mlx_memory_ceiling_after_wired_claim(
    effective_mlx_memory_ceiling_bytes: u64,
    claimed_wired_residency_budget_bytes: Option<u64>,
) -> u64 {
    match claimed_wired_residency_budget_bytes {
        Some(claimed_wired_residency_budget_bytes) => {
            let plannable_wired_residency_bytes =
                plannable_wired_residency_bytes(claimed_wired_residency_budget_bytes);
            if plannable_wired_residency_bytes < effective_mlx_memory_ceiling_bytes {
                plannable_wired_residency_bytes
            } else {
                effective_mlx_memory_ceiling_bytes
            }
        }
        None => effective_mlx_memory_ceiling_bytes,
    }
}

/// Share of the wired-residency budget kept free of MLX planning, in parts
/// per thousand (25 = 2.5%).
const METAL_WORKING_SET_MARGIN_PARTS_PER_THOUSAND: u64 = 25;

/// Wired residency the MLX residency planner may fill: the claimed budget
/// minus a 2.5% Metal working-set margin.
///
/// MLX `active_memory` does not count everything Metal charges against the
/// wired budget: allocator-cache buffers, driver command-buffer scratch, and
/// other GPU clients sharing the system-wide wired pool. A planner that packs
/// MLX arrays to the raw claim therefore fails the command buffer with
/// `kIOGPUCommandBufferCallbackErrorOutOfMemory`. Measured 2026-10-24 on a
/// 48 GB machine with a 40,200,896,512-byte claim: a 122B decode peaked at
/// 40,051,177,002 MLX bytes (99.6% of the claim) and the worker died, while
/// the same model at 37.4 GB peaks (93%) never failed. Upstream MLX makes the
/// same judgement, starting cache garbage collection at 95% of Metal's
/// recommended working set. The margin used to exist only by accident inside
/// an oversized 1 GB context-reserve floor; it lives here, named, so the
/// context reserve can stay exact.
#[must_use]
pub const fn plannable_wired_residency_bytes(claimed_wired_residency_budget_bytes: u64) -> u64 {
    claimed_wired_residency_budget_bytes.saturating_sub(
        claimed_wired_residency_budget_bytes / 1_000 * METAL_WORKING_SET_MARGIN_PARTS_PER_THOUSAND,
    )
}

/// The highest ceiling a loaded model may be raised to so it can serve at
/// its floor: the maximum the user configured, when it is below what this Mac
/// allows, or no bound at all when it is not.
///
/// The decision uses the configured value before Metal's wired-residency clamp
/// is applied: a user limit between the plannable wired residency and physical
/// memory is still the user's, and a raise must never pass it. `None` (no
/// configured maximum) and a maximum at or above the machine both mean the
/// user set no limit, so only the Mac itself bounds the model.
#[must_use]
pub const fn floor_raise_limit_bytes(
    configured_mlx_memory_ceiling_bytes: Option<u64>,
    machine_mlx_memory_ceiling_bytes: u64,
) -> u64 {
    match configured_mlx_memory_ceiling_bytes {
        Some(configured_mlx_memory_ceiling_bytes)
            if configured_mlx_memory_ceiling_bytes < machine_mlx_memory_ceiling_bytes =>
        {
            configured_mlx_memory_ceiling_bytes
        }
        _ => u64::MAX,
    }
}

/// The ceiling to serve a loaded model at when its floor is above the
/// effective ceiling, or `None` when the effective ceiling stands.
///
/// A model's floor is the least memory its leanest execution needs. Refusing a
/// model only because the Mac is small is something Astronomical never does
/// (#1902), so when the floor is above the effective ceiling the model is
/// served at its floor, unless that would pass a limit the user set (see
/// [`floor_raise_limit_bytes`]), in which case the ceiling stands and the
/// engine refuses as before. Memory above Metal's wired claim is not wired, so
/// whether macOS pages it or a command buffer fails on a small Mac is left to
/// MLX and the system. Only the "nothing fits, not even the floor" outcome
/// changes; requests are still sized against the effective ceiling, which
/// keeps oversized prefills refused on large Macs (#1297).
#[must_use]
pub const fn floor_raised_ceiling_bytes(
    effective_mlx_memory_ceiling_bytes: u64,
    model_floor_bytes: u64,
    floor_raise_limit_bytes: u64,
) -> Option<u64> {
    if model_floor_bytes > effective_mlx_memory_ceiling_bytes
        && model_floor_bytes <= floor_raise_limit_bytes
    {
        Some(model_floor_bytes)
    } else {
        None
    }
}
