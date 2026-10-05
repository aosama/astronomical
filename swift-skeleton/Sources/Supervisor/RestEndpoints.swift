// RestEndpoints.swift — Supervisor
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 2): apps/supervisor openai_* REST (Representational
// State Transfer) endpoint handlers, composed over RestContract shapes.
//
// Carried contracts:
// - Endpoint handlers compose the four shared failure codes from
//   RestContract; no endpoint-local error vocabularies.
// - Semantics pinned by issues #638, #769, and #772 carry over unchanged.
// - No backward compatibility is owed on the REST surface (no downstream
//   consumers), but contract tests still pin the wire behavior.
