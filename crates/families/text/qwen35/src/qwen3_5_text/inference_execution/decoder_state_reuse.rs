//! The engine's part of an `astro_ssd_prompt_cache` restore: room
//! leased from the memory governor for the blocks a restore loads beside the
//! stored state they become. The rest of the restore lives in
//! `text_generation::generation_loop::astro_ssd_prompt_cache_restore`.

use crate::InferenceEngineError;
use text_generation::generation_loop::astro_ssd_prompt_cache_restore::TextAstroSsdPromptCacheRestoreHooks;

use super::Qwen3_5EngineState;

impl TextAstroSsdPromptCacheRestoreHooks for Qwen3_5EngineState {
    /// Leases room from the memory governor for the blocks a restore loads
    /// beside the stored state they become, and returns the kept weight
    /// bytes given back for it. A restore that fails leaves the lease to
    /// the end of the request, which gives it back.
    fn lease_restore_workspace(
        &self,
        restore_workspace_bytes: usize,
    ) -> Result<u64, InferenceEngineError> {
        let model = self.core.loaded_model()?;
        Ok(model
            .govern_restore_workspace(u64::try_from(restore_workspace_bytes).unwrap_or(u64::MAX)))
    }

    /// The restored blocks have joined the stored state: their room goes
    /// back to the kept weights.
    fn give_back_restore_workspace(&self) {
        if let Some(model) = self.core.model.as_ref() {
            model.govern_restore_workspace_end();
        }
    }
}
