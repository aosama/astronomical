//! What the shared execution steps ask of a Qwen3.5 text model.
//!
//! The resident and streaming models differ in how they hold expert weights,
//! so each answers these questions its own way; the steps that ask them are
//! shared. The trait grows as more of the generation loop becomes shared.

use ipc_protocol::ExpertMemoryMode;
use memory_policy::ledger::LedgerReading;

use crate::inference_engine::contract::ExpertResidencyTelemetry;
use crate::memory::telemetry::MlxActiveMemoryBreakdown;
use crate::qwen3_5_core::model::memory_breakdown::Qwen3_5MemoryOwnership;

pub trait Qwen3_5TextModel: Qwen3_5MemoryOwnership {
    /// How expert weights are held as a reply token is emitted. The resident
    /// model always holds them whole; the streaming model reports what it
    /// observed for the forward that produced the token.
    fn observed_expert_memory_mode(&self) -> ExpertMemoryMode;

    /// The expert residency claim derived from one memory breakdown, so the
    /// telemetry and the claim describe the same instant.
    fn expert_residency_telemetry_for_breakdown(
        &self,
        active_memory_breakdown: &MlxActiveMemoryBreakdown,
    ) -> ExpertResidencyTelemetry;

    /// The memory governor's latest ledger reading, when it has one.
    fn latest_memory_ledger_reading(&self) -> Option<LedgerReading>;

    /// The MLX memory reading telemetry reports. The streaming model raises
    /// its peak to the largest prompt layer it measured, which MLX's own
    /// counter misses when layers are evaluated one at a time.
    fn memory_snapshot_for_telemetry(
        &self,
    ) -> Result<runtime_integration::MlxMemorySnapshot, crate::InferenceEngineError>;
}
