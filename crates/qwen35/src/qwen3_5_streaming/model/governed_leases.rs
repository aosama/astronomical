//! Room the memory governor leases for unusual needs, and stored state that
//! grows outside a measured step (issue #2221).
//!
//! A lease is room taken out of the kept-weight capacity for as long as
//! something temporary needs it: snapshots waiting to be written to the
//! `astro_ssd_prompt_cache`, blocks read back from it. Granting one lowers the
//! capacity, the kept weights give back what that displaces, and giving the
//! lease back raises the capacity again.

use memory_policy::ledger::{LeasePurpose, MemoryGovernor};

use super::Qwen3_5Model;

impl Qwen3_5Model {
    /// Publishing astro SSD prompt cache blocks needs `workspace_bytes` more than is
    /// free. Leases them until the next forward completes, has the astro expert cache
    /// give back that much, and returns the allocator's freed buffers to
    /// the system so the retry finds the room.
    pub fn govern_publication_workspace(
        &self,
        workspace_bytes: u64,
    ) -> Result<(), crate::InferenceEngineError> {
        let capacity = {
            let mut memory_governor = self.governed_memory.governor.borrow_mut();
            let Some(memory_governor) = memory_governor.as_mut() else {
                return Ok(());
            };
            if let Some(earlier_lease) = self.governed_memory.publication_workspace_lease.take() {
                let _ = memory_governor.give_back_lease(earlier_lease);
            }
            let (publication_lease, capacity) = memory_governor.grant_lease(
                LeasePurpose::AstroSsdPromptCachePublication,
                workspace_bytes,
            );
            self.governed_memory
                .publication_workspace_lease
                .set(Some(publication_lease));
            capacity
        };
        self.apply_kept_weight_capacity(capacity);
        self.runtime()
            .synchronize_gpu_stream_and_clear_mlx_buffer_cache()
            .map_err(crate::qwen3_5_streaming::inference_execution::qwen3_5_runtime_error)
    }

    /// A forward measured layer by layer still ends the publication its
    /// workspace lease covered.
    pub(super) fn give_back_publication_workspace_after_forward(&self) {
        let capacity = {
            let mut memory_governor = self.governed_memory.governor.borrow_mut();
            let Some(memory_governor) = memory_governor.as_mut() else {
                return;
            };
            self.give_back_publication_workspace(memory_governor);
            memory_governor.kept_weight_capacity()
        };
        self.apply_kept_weight_capacity(capacity);
    }

    /// Gives a publication lease back once the work it covered is over.
    pub(super) fn give_back_publication_workspace(&self, memory_governor: &mut MemoryGovernor) {
        if let Some(publication_lease) = self.governed_memory.publication_workspace_lease.take() {
            let _ = memory_governor.give_back_lease(publication_lease);
        }
    }

    /// Blocks read back from the `astro_ssd_prompt_cache` need `workspace_bytes` while
    /// they are loaded, beside the stored state they become. Leases that
    /// room until the restore is over and returns the kept weight bytes
    /// given back for it.
    pub fn govern_restore_workspace(&self, workspace_bytes: u64) -> u64 {
        let kept_weight_bytes_before_lease = self.reclaimable_retained_payload_bytes();
        let capacity = {
            let mut memory_governor = self.governed_memory.governor.borrow_mut();
            let Some(memory_governor) = memory_governor.as_mut() else {
                return 0;
            };
            self.give_back_restore_workspace_lease(memory_governor);
            let (restore_lease, capacity) = memory_governor
                .grant_lease(LeasePurpose::AstroSsdPromptCacheRestore, workspace_bytes);
            self.governed_memory
                .restore_workspace_lease
                .set(Some(restore_lease));
            capacity
        };
        self.apply_kept_weight_capacity(capacity);
        kept_weight_bytes_before_lease.saturating_sub(self.reclaimable_retained_payload_bytes())
    }

    /// The restore is over, whether it succeeded or not: its room goes back.
    pub fn govern_restore_workspace_end(&self) {
        let capacity = {
            let mut memory_governor = self.governed_memory.governor.borrow_mut();
            let Some(memory_governor) = memory_governor.as_mut() else {
                return;
            };
            self.give_back_restore_workspace_lease(memory_governor);
            memory_governor.kept_weight_capacity()
        };
        self.apply_kept_weight_capacity(capacity);
    }

    pub(super) fn give_back_restore_workspace_lease(&self, memory_governor: &mut MemoryGovernor) {
        if let Some(restore_lease) = self.governed_memory.restore_workspace_lease.take() {
            let _ = memory_governor.give_back_lease(restore_lease);
        }
    }

    /// The request's stored state is about to reach `context_bytes`, by
    /// tokens fed back into it between measured steps. The context account
    /// grows to that before the tokens run, so the kept weights have made
    /// room first.
    pub fn govern_context_extent(&self, context_bytes: u64) {
        let capacity = {
            let mut memory_governor = self.governed_memory.governor.borrow_mut();
            let Some(memory_governor) = memory_governor.as_mut() else {
                return;
            };
            let context_growth_bytes =
                context_bytes.saturating_sub(memory_governor.accounts().context_bytes);
            if context_growth_bytes == 0 {
                return;
            }
            memory_governor.grow_context(context_growth_bytes)
        };
        self.apply_kept_weight_capacity(capacity);
    }
}
