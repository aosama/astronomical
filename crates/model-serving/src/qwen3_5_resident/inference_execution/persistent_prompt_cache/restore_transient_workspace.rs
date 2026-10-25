//! Transient workspace arithmetic for one persistent prompt-cache restore.

/// Upper bound of restore's transient workspace in bytes. The concatenated
/// key-value blocks and the recurrent snapshot never coexist — the snapshot
/// loads only after the block set materializes and releases — so the
/// workspace is the larger of the two totals, saturating into usize.
pub(super) fn restore_transient_workspace_bytes(
    total_sequence_block_file_size_bytes: u64,
    recurrent_snapshot_file_size_bytes: u64,
) -> usize {
    usize::try_from(total_sequence_block_file_size_bytes.max(recurrent_snapshot_file_size_bytes))
        .unwrap_or(usize::MAX)
}

#[cfg(test)]
#[path = "restore_transient_workspace_tests.rs"]
mod restore_transient_workspace_tests;
