// Flux2KleinImageEngine.swift — ModelServing.Flux2Klein.Engine
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 3): crates/model-serving/src/flux2_klein/engine/* —
// components, lifecycle, mlx_components (image generation engine), plus
// crates/config/src/model_discovery/
// {flux2_klein,flux2_klein_documents}.rs discovery shapes.
//
// Carried contracts:
// - Bounded artifact reads stay bounded (JSON 4 MiB, text-encoder index
//   32 MiB, sidecar 64 MiB, license 64 KiB).
// - Diffusion-step timings feed the switchable performance attribution log.
