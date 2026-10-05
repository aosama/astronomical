// ImageGenerationProtocol.swift — IpcProtocol.ImageGeneration
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 1): crates/ipc-protocol/src/image_generation/* plus
// the flat image_generation.rs and png_validation.rs at the crate root.
//
// Carried contracts:
// - Image payloads crossing the IPC (Inter-Process Communication) boundary
//   stay size-bounded; PNG (Portable Network Graphics) validation happens
//   at the protocol edge, not inside the worker.
