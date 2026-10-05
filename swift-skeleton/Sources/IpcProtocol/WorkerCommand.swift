// WorkerCommand.swift — IpcProtocol
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 1): crates/ipc-protocol command types sent from the
// supervisor to inference workers (model load, generation requests, and the
// rest of the command surface).
//
// Carried contracts:
// - Command and event types stay paired with WorkerEvent so every request
//   shape has a matching completion shape.
// - New commands get contract tests before any producer or consumer ships.
