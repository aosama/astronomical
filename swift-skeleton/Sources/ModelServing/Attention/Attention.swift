// Attention.swift — ModelServing.Attention
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 3): crates/model-serving/src/attention/* —
// mlx_sliding_window_mask, sliding_window_visibility, yarn_frequencies.
//
// Carried contracts:
// - Sliding-window visibility and YaRN (Yet another RoPE extensioN)
//   frequency tables compute through MLX stock operations; custom masks
//   ship only with a numerically measured win.
// - Mask math stays shared across families rather than duplicated per
//   family module.
