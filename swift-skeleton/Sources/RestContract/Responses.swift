// Responses.swift — RestContract
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 2): crates/rest-contract responses-endpoint request
// and response shapes (the responses API surface).
//
// Carried contracts:
// - Streaming event ordering stays explicit in the contract so the supervisor
//   and any client render from the same sequence definition.
// - Failure shapes reuse the four shared failure codes rather than growing
//   endpoint-specific error enums.
