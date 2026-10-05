// ChatCompletions.swift — RestContract
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 2): crates/rest-contract chat-completions request and
// response shapes, as served by apps/supervisor openai_* endpoints.
//
// Carried contracts:
// - The four failure codes and their semantics stay exactly as pinned by
//   issues #638, #769, and #772.
// - Wire behavior is pinned by contract tests even though no backward
//   compatibility is owed to downstream consumers.
// - Temperature is never defaulted to 0; the provider decides.
