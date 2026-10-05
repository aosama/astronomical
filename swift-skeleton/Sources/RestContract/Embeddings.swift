// Embeddings.swift — RestContract
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 2): crates/rest-contract embeddings request and
// response shapes.
//
// Carried contracts:
// - Vector payloads stay bounded by the same frame cap rules as the rest of
//  the IPC (Inter-Process Communication) surface when relayed internally.
// - Request validation errors use the shared failure codes.
