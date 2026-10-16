//! Resolution of the effective MLX memory ceiling from the user's
//! configuration and the machine bound.
//!
//! One source of truth governs the maximum: physical RAM minus the
//! operating-system survival reserve. The user's configured value may cap
//! below that bound; with no configuration the machine bound itself is the
//! ceiling. No graphics-memory recommendation from the kernel or the device
//! participates in the bound.

/// End-user-facing memory values use the gigabyte convention Apple uses to
/// market the machine: 1 GB = 1,073,741,824 bytes (2³⁰), so a
/// 51,539,607,552-byte machine is the "48 GB" on the purchase sticker.
const BYTES_PER_MARKETED_GIGABYTE: u64 = 1_073_741_824;
/// What macOS keeps for itself so the machine stays usable while a model
/// holds the rest of RAM.
const OPERATING_SYSTEM_SURVIVAL_RESERVE_BYTES: u64 = 8 * BYTES_PER_MARKETED_GIGABYTE;
/// Smallest machine ceiling so very small computers remain servable instead
/// of resolving to a zero ceiling that rejects every configuration.
const MINIMUM_MACHINE_MLX_MEMORY_CEILING_BYTES: u64 = BYTES_PER_MARKETED_GIGABYTE;

/// Resolves the machine MLX memory ceiling from the physical memory size.
///
/// The ceiling is the total RAM minus the 8 GB operating-system survival
/// reserve, floored at 1 GB. Example: a 51,539,607,552-byte machine (sold
/// as 48 GB) resolves to 42,949,672,960 bytes — the 40 GB on the same
/// sticker scale.
#[must_use]
pub const fn machine_ceiling_from_physical_ram_bytes(physical_ram_bytes: u64) -> u64 {
    let ceiling_after_reserve_bytes =
        physical_ram_bytes.saturating_sub(OPERATING_SYSTEM_SURVIVAL_RESERVE_BYTES);
    if ceiling_after_reserve_bytes < MINIMUM_MACHINE_MLX_MEMORY_CEILING_BYTES {
        MINIMUM_MACHINE_MLX_MEMORY_CEILING_BYTES
    } else {
        ceiling_after_reserve_bytes
    }
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
#[must_use]
pub const fn effective_mlx_memory_ceiling_after_wired_claim(
    effective_mlx_memory_ceiling_bytes: u64,
    claimed_wired_residency_budget_bytes: Option<u64>,
) -> u64 {
    match claimed_wired_residency_budget_bytes {
        Some(claimed_wired_residency_budget_bytes)
            if claimed_wired_residency_budget_bytes < effective_mlx_memory_ceiling_bytes =>
        {
            claimed_wired_residency_budget_bytes
        }
        _ => effective_mlx_memory_ceiling_bytes,
    }
}
