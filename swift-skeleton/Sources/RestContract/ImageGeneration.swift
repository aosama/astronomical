// ImageGeneration.swift — RestContract
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 2): crates/rest-contract image-generation request and
// response shapes (served by the image families Qwen-Image-2.1, FLUX.2
// Klein).
//
// Carried contracts:
// - End-user-facing image sizes stay decimal SI (1 GB = 1,000,000,000
//   bytes).
// - Generated-artifact metadata reports sizes in decimal units consistently
//   across REST (Representational State Transfer) responses and CLI output.
