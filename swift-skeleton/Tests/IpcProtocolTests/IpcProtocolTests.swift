// IpcProtocolTests.swift — IpcProtocolTests
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Test target for Sources/IpcProtocol (wave 1). Tests mirror the source
// package paths.
//
// Carried contracts:
// - Round-trip cases pin the wire encoding byte-for-byte; the Swift menu
//   app's decoders are the downstream consumer these cases protect.
// - Frame cases cover the 32 MiB cap boundary and malformed-frame
//   rejection.
// - Every test carries a built-in timeout capped at 120 seconds.
