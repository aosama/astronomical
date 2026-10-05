// Transformer.swift — ModelServing.QwenImage21.Transformer
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 3): crates/model-serving/src/qwen_image_21/
// transformer/* (the diffusion transformer blocks; rope, modulation, norm,
// and mlx_math stay flat at the family root, as in Rust).
//
// Carried contracts:
// - Transformer math runs through MLX stock compile-fused layers; custom
//   kernels only with a numerically measured win.
// - GPU (Graphics Processing Unit) dispatches are combined wherever the
//   stock API allows.
