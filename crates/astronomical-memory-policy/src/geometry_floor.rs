use crate::budget::ram_values::MlxRamBudgetModelGeometry;

/// Bootstrap context reserve a cold engine holds before any live measurement (1 GB SI).
const GEOMETRY_FLOOR_BOOTSTRAP_CONTEXT_RESERVE_BYTES: u64 = 1_000_000_000;

/// Smallest ceiling under which SSD streaming can still serve the next request.
///
/// The floor is derived from validated artifact geometry once at load and is
/// deliberately independent of the ceiling that produced the current load:
/// model core plus the cold-start context reserve plus the streaming
/// workspace (one complete expert layer) plus one exact routed expert page
/// for decode. Retained expert payload never counts because the pager can
/// reclaim it.
#[must_use]
pub const fn geometry_safe_minimum_mlx_memory_ceiling_bytes(
    model_geometry: &MlxRamBudgetModelGeometry,
) -> u64 {
    model_geometry
        .model_core_payload_bytes
        .saturating_add(GEOMETRY_FLOOR_BOOTSTRAP_CONTEXT_RESERVE_BYTES)
        .saturating_add(model_geometry.largest_complete_expert_layer_bytes)
        .saturating_add(model_geometry.largest_routed_expert_page_bytes)
}
