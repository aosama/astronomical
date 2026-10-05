// MtpVerify.swift — ModelServing.Qwen3_5.MtpVerify
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 3): crates/model-serving/src/qwen3_5/mtp_verify/*
// (verification of multi-token-prediction draft tokens).
//
// Carried contracts:
// - Verified acceptance is all-or-nothing per draft block; a rejected
//   block rolls the decoder state back to the accepted prefix.
