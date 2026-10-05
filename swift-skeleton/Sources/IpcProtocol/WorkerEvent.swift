// WorkerEvent.swift — IpcProtocol
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 1): crates/ipc-protocol event types sent from
// inference workers back to the supervisor (generation progress, errors,
// lifecycle events).
//
// Carried contracts:
// - Events stay paired with WorkerCommand shapes; no orphan completions.
// - The already-Swift menu app decodes a subset of this surface — any new
//   event carries a decoder update in the same change.
