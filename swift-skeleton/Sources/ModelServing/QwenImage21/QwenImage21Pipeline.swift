// QwenImage21Pipeline.swift — ModelServing.QwenImage21
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 3): crates/model-serving qwen_image_21 image
// generation pipeline, plus crates/config/src/model_discovery/
// {qwen_image_21,qwen_image_21_documents}.rs discovery shapes.
//
// Carried contracts:
// - Bounded artifact reads stay bounded (JSON 4 MiB, component index
//   32 MiB, sidecar 64 MiB, README 256 KiB).
// - End-user-facing image and artifact sizes stay decimal SI.
