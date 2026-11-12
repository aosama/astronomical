//! What a speech engine is built with before it loads: which model, where
//! its files are, and the MLX memory limits the worker grants it.

use std::path::PathBuf;

use astronomical_config::SpeechModelCapabilities;
use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};
use astronomical_speech_contract::SpeechFailure;

use super::speech_memory::SpeechMemoryAccount;
use crate::speech_engine::SpeechEngineLoadResult;

pub(crate) struct SpeechEngineSettings {
    pub(crate) model_id: String,
    pub(crate) model_directory: PathBuf,
    pub(crate) effective_mlx_memory_ceiling_bytes: usize,
    pub(crate) allocator_cache_memory_limit_bytes: usize,
}

impl SpeechEngineSettings {
    /// The engine's own MLX runtime under the granted limits.
    pub(crate) fn initialize_runtime(&self) -> Result<MlxRuntime, SpeechFailure> {
        let memory_limits = MlxMemoryLimits::new(
            self.effective_mlx_memory_ceiling_bytes,
            self.allocator_cache_memory_limit_bytes,
        )
        .map_err(|limits_error| SpeechFailure::fatal_execution(limits_error.to_string()))?;
        MlxRuntime::initialize(memory_limits)
            .map_err(|runtime_error| SpeechFailure::fatal_execution(runtime_error.to_string()))
    }

    /// What the loaded model offers, as discovery read it from the folder:
    /// the voices a caller was told about are the ones the engine accepts.
    /// `model_sample_rate_hertz` is the rate the loaded decoder produces; a
    /// folder whose advertised rate differs would play at the wrong speed.
    pub(crate) fn load_result(
        &self,
        speech_capabilities: Option<SpeechModelCapabilities>,
        model_sample_rate_hertz: u32,
    ) -> Result<SpeechEngineLoadResult, SpeechFailure> {
        let speech_capabilities = speech_capabilities.ok_or_else(|| {
            SpeechFailure::fatal_execution("the speech model folder has no voice it can speak in")
        })?;
        if speech_capabilities.sample_rate_hertz != model_sample_rate_hertz {
            return Err(SpeechFailure::fatal_execution(format!(
                "the folder advertises {} Hz but its decoder produces {model_sample_rate_hertz} Hz",
                speech_capabilities.sample_rate_hertz
            )));
        }
        Ok(SpeechEngineLoadResult {
            model_id: self.model_id.clone(),
            sample_rate_hertz: speech_capabilities.sample_rate_hertz,
            voice_names: speech_capabilities.voice_names,
            default_voice_name: speech_capabilities.default_voice_name,
            supports_speed: speech_capabilities.supports_speed,
        })
    }

    pub(crate) fn memory_account(&self, resident_weight_bytes: u64) -> SpeechMemoryAccount {
        SpeechMemoryAccount::new(
            self.effective_mlx_memory_ceiling_bytes as u64,
            resident_weight_bytes,
        )
    }
}
