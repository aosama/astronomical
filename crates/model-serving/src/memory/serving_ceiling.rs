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

/// Resolves the effective ceiling under the configured and machine limits.
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
