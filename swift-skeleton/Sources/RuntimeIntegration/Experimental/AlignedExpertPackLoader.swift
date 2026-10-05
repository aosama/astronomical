// AlignedExpertPackLoader.swift — RuntimeIntegration.Experimental
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 3): crates/runtime-integration/src/experimental/
// aligned_expert_pack_loader.rs.
//
// Carried contracts:
// - Experimental status must survive the migration explicitly: this loader
//   stays opt-in through configuration, never a silent default.
// - Aligned pack loads feed the switchable performance attribution log so
//   their cost stays visible next to the stock loading path.
