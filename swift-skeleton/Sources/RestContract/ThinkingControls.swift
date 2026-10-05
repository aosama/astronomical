// ThinkingControls.swift — RestContract
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 2): crates/rest-contract thinking-controls shapes
// (reasoning effort and related controls across chat and responses).
//
// Carried contracts:
// - Thinking controls stay a first-class part of the wire contract, not an
//   extension bag; both chat-completions and responses surfaces expose the
//   same control set.
