//! The memory ledger's reading as the worker publishes it (issue #2221).

use astronomical_ipc_protocol::WorkerMemoryLedgerSnapshot;
use astronomical_memory_policy::ledger::LedgerReading;

/// Flattens one ledger reading into the protocol's snapshot.
pub(crate) const fn worker_memory_ledger_snapshot(
    ledger_reading: LedgerReading,
) -> WorkerMemoryLedgerSnapshot {
    let accounts = ledger_reading.accounts;
    let held_memory = ledger_reading.held_memory;
    let reconciliation = ledger_reading.reconciliation;
    let (system_wired_bytes, system_wire_limit_bytes) = match ledger_reading.system_wired_reading {
        Some(system_wired_reading) => system_wired_reading,
        None => (0, 0),
    };
    WorkerMemoryLedgerSnapshot {
        limit_bytes: accounts.limit_bytes,
        core_weight_bytes: accounts.core_weight_bytes,
        context_bytes: accounts.context_bytes,
        mlx_working_buffer_bytes: accounts.mlx_working_buffer_bytes,
        mlx_buffer_cache_bytes: accounts.mlx_buffer_cache_bytes,
        system_wired_reserve_bytes: accounts.system_wired_reserve_bytes,
        system_wired_bytes,
        system_wire_limit_bytes,
        stream_slot_bytes: accounts.stream_slot_bytes,
        lease_bytes: accounts.lease_bytes,
        margin_bytes: accounts.margin_bytes,
        kept_weight_capacity_bytes: accounts.kept_weight_capacity_bytes,
        unassigned_bytes: accounts.unassigned_bytes,
        shortfall_bytes: accounts.shortfall_bytes,
        mlx_working_buffers_are_measured: accounts.mlx_working_buffers_are_measured,
        held_context_bytes: held_memory.context_bytes,
        held_kept_weight_bytes: held_memory.kept_weight_bytes,
        held_stream_slot_bytes: held_memory.stream_slot_bytes,
        expected_active_bytes: reconciliation.expected_active_bytes,
        unaccounted_bytes: reconciliation.unaccounted_bytes,
        overcounted_bytes: reconciliation.overcounted_bytes,
        kept_over_capacity_bytes: reconciliation.kept_over_capacity_bytes,
        step_peak_beyond_held_bytes: ledger_reading.step_peak_beyond_held_bytes,
        observed_step_count: ledger_reading.observed_step_count,
        largest_unaccounted_bytes: ledger_reading.largest_unaccounted_bytes,
        largest_overcounted_bytes: ledger_reading.largest_overcounted_bytes,
        largest_kept_over_capacity_bytes: ledger_reading.largest_kept_over_capacity_bytes,
        largest_reading_step_peak_beyond_held_bytes: ledger_reading
            .largest_reading_step_peak_beyond_held_bytes,
        largest_writing_step_peak_beyond_held_bytes: ledger_reading
            .largest_writing_step_peak_beyond_held_bytes,
        probed_mlx_working_buffer_bytes: ledger_reading.probed_mlx_working_buffer_bytes,
        lowest_kept_share_while_reading_permille: ledger_reading
            .lowest_kept_share_while_reading_permille,
    }
}

#[cfg(test)]
#[path = "memory_ledger_snapshot_tests.rs"]
mod tests;
