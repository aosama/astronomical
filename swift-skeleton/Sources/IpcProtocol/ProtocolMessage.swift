// ProtocolMessage.swift — IpcProtocol
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 1): crates/ipc-protocol message and model
// configuration types, including worker_model_configuration.
//
// Carried contracts:
// - This module is the single authority for the wire protocol; the Swift
//   menu app's decoders must move in lockstep with any change here — the
//   contract tests are the guard against drift between the two.
// - Wave 2's strangler seam depends on these types staying byte-compatible
//   while the supervisor and CLI swap implementations underneath.
