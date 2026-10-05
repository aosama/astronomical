// KernelCapability.swift — ModelServing.KernelCapability
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 3): crates/model-serving/src/kernel_capability/* —
// fused_expert_decode_probe, gated_delta_probes,
// sorted_expert_weighted_sum_probe, target_verification_probes.
//
// Carried contracts:
// - Custom or fused kernels are used only after the probe verifies the
//   target supports them — and only with a numerically measured win over
//   the MLX stock path.
// - Probe results cache per device, so capability checks never run per
//   token generation step.
