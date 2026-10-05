// LagunaText.swift — ModelServing.Laguna.Text
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 3): crates/model-serving/src/laguna/text/* including
// its output_parser submodule (text inference path and output parsing).
//
// Carried contracts:
// - The output parser tolerates partial tokens at stream boundaries; it
//   never blocks the stream waiting for a complete unit.
// - The Romeo and Juliet fixture is the acceptance input for text-family
//   journeys; no random text or tokens.
