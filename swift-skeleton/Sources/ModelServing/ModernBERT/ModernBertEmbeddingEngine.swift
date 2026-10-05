// ModernBertEmbeddingEngine.swift — ModelServing.ModernBERT
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 3): crates/model-serving modernbert embedding family
// runtime, plus crates/config/src/model_discovery/modernbert.rs discovery
// shapes.
//
// Carried contracts:
// - Embedding responses relay through the bounded frame transport; vector
//   batch sizes respect the 32 MiB per-frame cap.
// - Bounded artifact reads stay bounded per the discovery module's caps.
