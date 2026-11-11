//! Phonon end to end: text chunks and a voice in, 24 kHz mono samples out.
//!
//! The voice is read once; each text chunk then starts from that state, is
//! read, and is spoken frame by frame until the end-of-speech countdown or
//! the chunk's frame budget stops it. The chunk's latents are decoded to
//! audio by Mimi.

use std::path::Path;

use astronomical_runtime_integration::MlxRuntime;
use mlx_bridge::{MlxArray, MlxDtype};

use super::artifact_files::{
    PHONON_CONFIGURATION_FILE_NAME, PHONON_TOKENIZER_FILE_NAME, PHONON_WEIGHTS_FILE_NAME,
};
use super::attention_history::AttentionHistory;
use super::configuration::PhononConfiguration;
use super::flow_language_model::FlowLanguageModel;
use super::latent_generation::{ChunkGenerationReport, EndOfSpeechCountdown, frame_budget};
use super::mimi_decoder::MimiDecoder;
use super::text_chunks::{TextChunk, prompt_chunks};
use super::text_tokenizer::PhononTokenizer;
use super::voice::PhononVoice;
use crate::speech_models::checkpoint_tensors::{
    CheckpointTensors, load_safetensors_file, read_text_artifact,
};
use crate::speech_models::speech_model_error::SpeechModelError;

/// The sampling temperature the reference runtime uses: its `Synth`
/// default and `generation_config.json` both say 0.3. `config.json` still
/// carries `"temp": 0.8`, a key the reference runtime no longer reads.
pub(crate) const PHONON_DEFAULT_TEMPERATURE: f32 = 0.3;

pub(crate) struct PhononSpeechModel {
    configuration: PhononConfiguration,
    tokenizer: PhononTokenizer,
    flow: FlowLanguageModel,
    mimi: MimiDecoder,
}

/// How one request samples.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PhononSampling {
    /// The noise's variance; 0 is deterministic. Not checked here: the
    /// engine always passes [`PHONON_DEFAULT_TEMPERATURE`], and a negative
    /// value would make every noise sample NaN.
    pub(crate) temperature: f32,
    /// Chunk `n` of a request uses `seed + n`, so the same seed gives the
    /// same audio for the same text.
    pub(crate) seed: u64,
}

impl PhononSpeechModel {
    pub(crate) fn load(
        runtime: &MlxRuntime,
        model_directory: &Path,
    ) -> Result<Self, SpeechModelError> {
        let configuration_json =
            read_text_artifact(&model_directory.join(PHONON_CONFIGURATION_FILE_NAME))?;
        let configuration = PhononConfiguration::parse(&configuration_json)?;
        let weights =
            load_safetensors_file(runtime, &model_directory.join(PHONON_WEIGHTS_FILE_NAME))?;
        let checkpoint = CheckpointTensors {
            runtime,
            tensors: &weights,
        };
        Ok(Self {
            tokenizer: PhononTokenizer::load(&model_directory.join(PHONON_TOKENIZER_FILE_NAME))?,
            flow: FlowLanguageModel::load(&checkpoint, &configuration)?,
            mimi: MimiDecoder::load(&checkpoint, &configuration.mimi)?,
            configuration,
        })
    }

    pub(crate) fn sample_rate_hertz(&self) -> u32 {
        self.configuration.sample_rate_hertz()
    }

    /// `text` as the chunks [`Self::speak_chunk`] speaks, at least one.
    pub(crate) fn prompt_chunks(&self, text: &str) -> Result<Vec<TextChunk>, SpeechModelError> {
        let chunks = prompt_chunks(&self.tokenizer, text)?;
        if chunks.is_empty() {
            return Err(SpeechModelError::Input(
                "there is no text to speak".to_owned(),
            ));
        }
        Ok(chunks)
    }

    /// The flow model's state after reading `voice`, which every chunk starts from.
    pub(crate) fn voice_history(
        &self,
        runtime: &MlxRuntime,
        voice: &PhononVoice,
    ) -> Result<AttentionHistory, SpeechModelError> {
        let mut voice_history = self.flow.empty_history();
        self.flow
            .prompt_voice(runtime, &mut voice_history, &voice.speaker_latents)?;
        Ok(voice_history)
    }

    /// One chunk's samples; chunk `n` samples with `seed + n`.
    pub(crate) fn speak_chunk(
        &self,
        runtime: &MlxRuntime,
        voice_history: &AttentionHistory,
        chunk: &TextChunk,
        chunk_index: usize,
        sampling: PhononSampling,
    ) -> Result<(Vec<f32>, ChunkGenerationReport), SpeechModelError> {
        let chunk_sampling = PhononSampling {
            seed: sampling.seed.wrapping_add(chunk_index as u64),
            ..sampling
        };
        let (latents, chunk_report) =
            self.generate_latents(runtime, voice_history, chunk, chunk_sampling)?;
        Ok((self.decode(runtime, &latents)?, chunk_report))
    }

    /// Normalised latents for one chunk, `[frames, latent width]`.
    fn generate_latents(
        &self,
        runtime: &MlxRuntime,
        voice_history: &AttentionHistory,
        chunk: &TextChunk,
        sampling: PhononSampling,
    ) -> Result<(MlxArray, ChunkGenerationReport), SpeechModelError> {
        let mut history = voice_history.branch()?;
        self.flow
            .prompt_text(runtime, &mut history, &chunk.token_ids)?;
        let frame_limit = frame_budget(
            chunk.token_ids.len(),
            self.configuration.mimi.frame_rate_hertz,
        );
        let all_noise = self.chunk_noise(runtime, frame_limit, sampling)?;
        let mut frames: Vec<MlxArray> = Vec::new();
        let mut previous_latent = self.flow.start_latent.retain()?;
        let mut countdown = EndOfSpeechCountdown::new(chunk.frames_after_end);
        for frame_index in 0..frame_limit {
            let noise = self.noise_row(runtime, &all_noise, frame_index)?;
            let frame = self
                .flow
                .next_frame(runtime, &mut history, &previous_latent, &noise)?;
            frames.push(frame.latent.retain()?);
            previous_latent = frame.latent;
            if countdown
                .should_stop(frame.end_of_speech_logit > self.configuration.end_of_speech_threshold)
            {
                break;
            }
        }
        let chunk_report = ChunkGenerationReport {
            frame_count: frames.len(),
            was_ended_by_end_of_speech: countdown.has_seen_end_of_speech(),
        };
        let frame_refs: Vec<&MlxArray> = frames.iter().collect();
        Ok((runtime.concatenate_axis(&frame_refs, 0)?, chunk_report))
    }

    /// Gaussian noise with variance `temperature`, one `[latent width]` row per frame.
    fn chunk_noise(
        &self,
        runtime: &MlxRuntime,
        frame_limit: usize,
        sampling: PhononSampling,
    ) -> Result<MlxArray, SpeechModelError> {
        let noise_key = runtime.random_key(sampling.seed)?;
        let noise_shape = [frame_limit as i32, self.configuration.flow.latent_width];
        let standard_noise =
            runtime.random_normal(&noise_shape, MlxDtype::Float32, 0.0, 1.0, &noise_key)?;
        Ok(runtime.multiply_scalar(&standard_noise, sampling.temperature.sqrt())?)
    }

    fn noise_row(
        &self,
        runtime: &MlxRuntime,
        all_noise: &MlxArray,
        frame_index: usize,
    ) -> Result<MlxArray, SpeechModelError> {
        let row_index = frame_index as i32;
        let latent_width = self.configuration.flow.latent_width;
        Ok(runtime.slice(
            all_noise,
            &[row_index, 0],
            &[row_index + 1, latent_width],
            &[1, 1],
        )?)
    }

    fn decode(
        &self,
        runtime: &MlxRuntime,
        normalised_latents: &MlxArray,
    ) -> Result<Vec<f32>, SpeechModelError> {
        let scaled = runtime.multiply(normalised_latents, &self.flow.latent_deviation)?;
        let latents = runtime.add(&scaled, &self.flow.latent_mean)?;
        let samples = runtime.array_to_vec_f32(&self.mimi.decode(runtime, &latents)?)?;
        let expected_sample_count =
            normalised_latents.shape()[0] as usize * self.configuration.samples_per_frame();
        if samples.len() != expected_sample_count {
            return Err(SpeechModelError::Artifact(format!(
                "Mimi gave {} samples where {expected_sample_count} were expected; the decoder does not match the checkpoint",
                samples.len()
            )));
        }
        Ok(samples)
    }
}
