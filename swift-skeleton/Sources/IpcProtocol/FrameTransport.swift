// FrameTransport.swift — IpcProtocol
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 1): crates/ipc-protocol frame encoding and transport
// (length-prefixed frames over local sockets).
//
// Carried contracts:
// - The per-frame cap stays 32 MiB (33,554,432 bytes); oversized frames are
//   rejected, never truncated or split silently.
// - Frame decoding stays total: malformed input produces a typed error, not
//   a crash.
