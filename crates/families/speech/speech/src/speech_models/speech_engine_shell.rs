//! The [`SpeechEngine`] every speech family runs behind: one runtime, one
//! memory account and one request slot, loaded and stepped the same way. A
//! family supplies only what differs, through [`SpeechFamily`]: how its model
//! loads, and how a request becomes its segments. Kokoro and Phonon each
//! carried their own copy of this shell before; a fix to the shell now reaches
//! both, and a new speech family writes none of it.

use std::path::{Path, PathBuf};

use config::SpeechModelCapabilities;
use memory_policy::live_memory_limit::{LiveMemoryLimitEngine, LiveMemoryLimitRefusal};
use memory_policy::telemetry::{MlxMemoryLimitAdjustment, MlxMemoryTelemetry};
use model_primitives::live_memory_ceiling::move_engine_ceiling;
use runtime_integration::MlxRuntime;
use speech_contract::{SpeechFailure, SpeechRequest};

use crate::speech_engine::{SpeechEngine, SpeechEngineLoadResult, SpeechEngineStep, StartedSpeech};
use crate::speech_models::checkpoint_tensors::weights_file_bytes;
use crate::speech_models::engine_settings::SpeechEngineSettings;
use crate::speech_models::speech_memory::SpeechMemoryAccount;
use crate::speech_models::speech_request_slot::{
    SegmentSpeaker, SpeechRequestSlot, ends_request, voice_for_checked_request,
};

/// What a speech family adds to the shared shell.
pub trait SpeechFamily: 'static {
    /// The family's loaded state beside the shared runtime: its model and
    /// whatever it keeps per voice.
    type FamilyParts: 'static;
    /// One unit a request is spoken in, one per step.
    type Segment: 'static;
    /// What every segment of one request is spoken with.
    type RequestContext: 'static;
    /// The name messages use, such as "Kokoro".
    const FAMILY_NAME: &'static str;
    /// The weights file whose size the memory account counts as resident.
    const WEIGHTS_FILE_NAME: &'static str;

    /// Loads the model on `runtime` and reports what discovery read from the
    /// folder, so the shell can check the two agree.
    fn load_family(
        runtime: &MlxRuntime,
        model_directory: &Path,
    ) -> Result<FamilyLoad<Self::FamilyParts>, SpeechFailure>;

    /// Turns one accepted request into its segments and shared context.
    /// `voice_name` has already been checked against the loaded voices.
    fn plan_speech(
        loaded: &mut LoadedSpeech<Self::FamilyParts>,
        speech_request: &SpeechRequest,
        voice_name: &str,
    ) -> Result<PlannedSpeech<Self::Segment, Self::RequestContext>, SpeechFailure>;
}

/// What a family's load produced.
pub struct FamilyLoad<FamilyParts> {
    pub family_parts: FamilyParts,
    pub speech_capabilities: Option<SpeechModelCapabilities>,
    pub model_sample_rate_hertz: u32,
}

/// One request, cut into the segments the engine speaks one per step.
pub struct PlannedSpeech<Segment, RequestContext> {
    pub segments: Vec<Segment>,
    pub request_context: RequestContext,
}

/// A loaded speech model: the shared runtime and accounting, and the family's
/// own parts.
pub struct LoadedSpeech<FamilyParts> {
    pub model_directory: PathBuf,
    pub runtime: MlxRuntime,
    pub load_result: SpeechEngineLoadResult,
    pub memory_account: SpeechMemoryAccount,
    pub family_parts: FamilyParts,
}

impl<FamilyParts> LoadedSpeech<FamilyParts> {
    /// Frees what one request left behind in the MLX allocator.
    pub fn clean_up_after_request(&mut self) {
        self.memory_account.clean_up_after_request(&self.runtime);
    }
}

pub struct SpeechEngineShell<Family: SpeechFamily> {
    settings: SpeechEngineSettings,
    loaded: Option<LoadedSpeech<Family::FamilyParts>>,
    request_slot: SpeechRequestSlot<Family::Segment, Family::RequestContext>,
}

impl<Family: SpeechFamily> SpeechEngineShell<Family> {
    pub fn new(
        model_id: impl Into<String>,
        model_directory: PathBuf,
        effective_mlx_memory_ceiling_bytes: usize,
        mlx_buffer_cache_limit_bytes: usize,
    ) -> Self {
        Self {
            settings: SpeechEngineSettings {
                model_id: model_id.into(),
                model_directory,
                effective_mlx_memory_ceiling_bytes,
                mlx_buffer_cache_limit_bytes,
            },
            loaded: None,
            request_slot: SpeechRequestSlot::empty(),
        }
    }

    pub fn loaded_mut(&mut self) -> Result<&mut LoadedSpeech<Family::FamilyParts>, SpeechFailure> {
        match self.loaded.as_mut() {
            Some(loaded) => Ok(loaded),
            None => Err(not_loaded_failure(Family::FAMILY_NAME)),
        }
    }
}

/// The failure a call before `load` returns, worded per family.
pub fn not_loaded_failure(family_name: &str) -> SpeechFailure {
    SpeechFailure::fatal_execution(format!("the {family_name} engine is not loaded"))
}

impl<Family> SpeechEngine for SpeechEngineShell<Family>
where
    Family: SpeechFamily,
    LoadedSpeech<Family::FamilyParts>: SegmentSpeaker<Family::Segment, Family::RequestContext>,
{
    fn load(&mut self) -> Result<SpeechEngineLoadResult, SpeechFailure> {
        let runtime = self.settings.initialize_runtime()?;
        let model_directory = self.settings.model_directory.clone();
        let family_load = Family::load_family(&runtime, &model_directory)?;
        let load_result = self.settings.load_result(
            family_load.speech_capabilities,
            family_load.model_sample_rate_hertz,
        )?;
        let memory_account = self.settings.memory_account(weights_file_bytes(
            &model_directory.join(Family::WEIGHTS_FILE_NAME),
        ));
        self.loaded = Some(LoadedSpeech {
            model_directory,
            runtime,
            load_result: load_result.clone(),
            memory_account,
            family_parts: family_load.family_parts,
        });
        Ok(load_result)
    }

    /// Refuses a second request before any segment is prepared, since
    /// preparing may be costly, then checks the voice before the family plans.
    fn start_speech(
        &mut self,
        speech_request: SpeechRequest,
    ) -> Result<StartedSpeech, SpeechFailure> {
        self.request_slot.ensure_idle()?;
        let loaded = self.loaded_mut()?;
        let voice_name = voice_for_checked_request(&speech_request, &loaded.load_result)?;
        let planned_speech = Family::plan_speech(loaded, &speech_request, &voice_name)?;
        let started_speech = StartedSpeech {
            voice_name,
            sample_rate_hertz: loaded.load_result.sample_rate_hertz,
            total_segments: planned_speech.segments.len(),
        };
        self.request_slot
            .start(planned_speech.segments, planned_speech.request_context)?;
        Ok(started_speech)
    }

    fn advance_speech(&mut self) -> Result<SpeechEngineStep, SpeechFailure> {
        let Some(loaded) = self.loaded.as_mut() else {
            return Err(not_loaded_failure(Family::FAMILY_NAME));
        };
        let speech_step = self.request_slot.advance(loaded);
        if ends_request(&speech_step) {
            loaded.clean_up_after_request();
        }
        speech_step
    }

    fn cancel_speech(&mut self) {
        self.request_slot.cancel();
        if let Some(loaded) = self.loaded.as_mut() {
            loaded.clean_up_after_request();
        }
    }

    fn take_post_cleanup_memory_telemetry(&mut self) -> Option<MlxMemoryTelemetry> {
        self.loaded
            .as_mut()?
            .memory_account
            .take_post_cleanup_memory_telemetry()
    }

    fn collect_mlx_memory_telemetry(&self) -> Option<MlxMemoryTelemetry> {
        let loaded = self.loaded.as_ref()?;
        loaded.memory_account.observe(&loaded.runtime)
    }
}

impl<Family> LiveMemoryLimitEngine for SpeechEngineShell<Family>
where
    Family: SpeechFamily,
    LoadedSpeech<Family::FamilyParts>: SegmentSpeaker<Family::Segment, Family::RequestContext>,
{
    fn update_mlx_memory_limit(
        &mut self,
        requested_mlx_memory_ceiling_bytes: u64,
    ) -> Result<MlxMemoryLimitAdjustment, LiveMemoryLimitRefusal> {
        // Taken between two segments as well as between requests (issue
        // #2681): what the floor reads as held already counts the request's
        // own memory, so a ceiling below it is refused and the speech goes on.
        let loaded_runtime = match self.loaded.as_mut() {
            Some(loaded) => Some(&mut loaded.runtime),
            None => None,
        };
        let applied_ceiling = move_engine_ceiling(
            loaded_runtime,
            &mut self.settings.effective_mlx_memory_ceiling_bytes,
            &mut self.settings.mlx_buffer_cache_limit_bytes,
            requested_mlx_memory_ceiling_bytes,
        )?;
        if let Some(loaded) = self.loaded.as_mut() {
            loaded
                .memory_account
                .set_effective_mlx_memory_ceiling_bytes(requested_mlx_memory_ceiling_bytes);
        }
        Ok(applied_ceiling.into_adjustment(
            requested_mlx_memory_ceiling_bytes,
            self.collect_mlx_memory_telemetry(),
        ))
    }
}

#[cfg(test)]
#[path = "speech_engine_shell_tests.rs"]
mod speech_engine_shell_tests;
