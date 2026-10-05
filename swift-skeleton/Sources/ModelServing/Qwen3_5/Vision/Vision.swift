// Vision.swift — ModelServing.Qwen3_5.Vision
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 3): crates/model-serving/src/qwen3_5/vision/*
// (vision tower for multimodal input).
//
// Carried contracts:
// - Input images are untrusted payloads: decode and resize stay bounded
//   and produce typed errors.
