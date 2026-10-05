// Flux2KleinTransformer.swift — ModelServing.Flux2Klein.Transformer
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 3): crates/model-serving/src/flux2_klein/
// transformer/* (the diffusion transformer blocks).
//
// Carried contracts:
// - Transformer math runs through MLX stock compile-fused layers; custom
//   kernels only with a numerically measured win.
// - GPU (Graphics Processing Unit) dispatches are combined wherever the
//   stock API allows.
