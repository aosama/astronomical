// ExpertPaging.swift — ModelServing.ExpertPaging
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 3): crates/model-serving/src/expert_paging/* —
// bounded_expert_reader, quantized_expert_manifest (+validation),
// safetensors_header, retained_expert_page_cache (+contract, reclamation),
// expert_cache_statistics, source_manifests.
//
// Carried contracts:
// - Expert paging is a first-class critical path: its timings feed the
//   switchable performance attribution log.
// - The retained page cache reclaims under the Memory subpackage's budget
//   authority — paging code never keeps its own memory arithmetic.
// - Expert reads from SSD (Solid State Drive) stay bounded per page and
//   validated against the quantized manifest.
