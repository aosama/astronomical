// K2HorizonMovaEngine.swift — ModelServing.K2HorizonMova.Engine
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 3): crates/model-serving/src/k2_horizon_mova/engine/*
// (execution) plus the flat cache_layout, expert_geometry, serving_settings
// at the family root, and
// crates/config/src/model_discovery/k2_horizon_mova.rs discovery shapes.
//
// Carried contracts:
// - Structural validity assertions come from config — never golden-master
//   constants coupled to one packaging variant.
// - Expert routing goes through the Memory subpackage's expert cache; no
//   family-local memory arithmetic.
