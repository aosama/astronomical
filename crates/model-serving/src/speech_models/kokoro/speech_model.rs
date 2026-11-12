//! Kokoro-82M's forward pass: one phoneme string and a voice in, 24 kHz mono
//! samples out.
//!
//! The steps follow Kokoro's `KModel.forward`: ALBERT reads the phonemes, the
//! predictor decides each phoneme's duration, the phoneme states are spread
//! over those frames, the predictor draws pitch and energy over the frames,
//! and the decoder and generator turn it all into a waveform.

use std::path::Path;

use astronomical_runtime_integration::MlxRuntime;
use mlx_bridge::MlxArray;

use super::acoustic_decoder::{AcousticDecoder, DecoderInputs};
use super::albert_encoder::AlbertEncoder;
use super::artifact_files::{KOKORO_CONFIGURATION_FILE_NAME, KOKORO_WEIGHTS_FILE_NAME};
use super::configuration::KokoroConfiguration;
use super::frame_alignment::frame_alignment;
use super::prosody_predictor::ProsodyPredictor;
use super::text_encoder::AcousticTextEncoder;
use super::utterance::{KokoroUtterance, validated_token_ids};
use super::voice_pack::UtteranceStyle;
use crate::speech_models::checkpoint_tensors::{
    CheckpointTensors, load_safetensors_file, read_text_artifact,
};
use crate::speech_models::speech_model_error::SpeechModelError;

pub(crate) const KOKORO_SAMPLE_RATE_HERTZ: u32 = 24_000;

/// One utterance's forward pass. Speaking keeps only the samples; a
/// reference comparison reads the durations and curves as well.
pub(crate) struct KokoroForwardPass {
    /// Frames held per token, the two markers included.
    pub(crate) frame_counts: Vec<i32>,
    /// `[2 * frames, 1]` pitch in hertz.
    pub(crate) pitch_curve: MlxArray,
    /// `[2 * frames, 1]` energy.
    pub(crate) energy_curve: MlxArray,
    /// `[600 * frames]` samples.
    pub(crate) samples: MlxArray,
}

pub(crate) struct KokoroSpeechModel {
    configuration: KokoroConfiguration,
    albert: AlbertEncoder,
    prosody: ProsodyPredictor,
    text_encoder: AcousticTextEncoder,
    decoder: AcousticDecoder,
}

/// Pitch, energy and text states on the utterance's frames.
struct AlignedProsody {
    /// Frames held per token, the two markers included.
    frame_counts: Vec<i32>,
    pitch_curve: MlxArray,
    energy_curve: MlxArray,
    aligned_text: MlxArray,
}

impl KokoroSpeechModel {
    pub(crate) fn load(
        runtime: &MlxRuntime,
        model_directory: &Path,
    ) -> Result<Self, SpeechModelError> {
        let configuration_json =
            read_text_artifact(&model_directory.join(KOKORO_CONFIGURATION_FILE_NAME))?;
        let configuration = KokoroConfiguration::parse(&configuration_json)?;
        let weights =
            load_safetensors_file(runtime, &model_directory.join(KOKORO_WEIGHTS_FILE_NAME))?;
        let checkpoint = CheckpointTensors {
            runtime,
            tensors: &weights,
        };
        Ok(Self {
            albert: AlbertEncoder::load(
                &checkpoint,
                configuration.albert_attention_head_count,
                configuration.albert_layer_count,
            )?,
            prosody: ProsodyPredictor::load(&checkpoint)?,
            text_encoder: AcousticTextEncoder::load(&checkpoint)?,
            decoder: AcousticDecoder::load(&checkpoint)?,
            configuration,
        })
    }

    /// The utterance's samples at [`KOKORO_SAMPLE_RATE_HERTZ`], nominally in
    /// [-1, 1]; audio encoding clamps them.
    pub(crate) fn speak(
        &self,
        runtime: &MlxRuntime,
        utterance: &KokoroUtterance<'_>,
    ) -> Result<Vec<f32>, SpeechModelError> {
        let forward_pass = self.forward(runtime, utterance)?;
        Ok(runtime.array_to_vec_f32(&forward_pass.samples)?)
    }

    /// The whole pass, with the durations and curves that led to the samples.
    pub(crate) fn forward(
        &self,
        runtime: &MlxRuntime,
        utterance: &KokoroUtterance<'_>,
    ) -> Result<KokoroForwardPass, SpeechModelError> {
        let token_ids =
            validated_token_ids(&self.configuration, utterance.phonemes, utterance.speed)?;
        let token_array = runtime.array_from_i32(&token_ids, &[token_ids.len() as i32])?;
        let style = utterance
            .voice
            .style_for(runtime, utterance.phonemes.chars().count())?;
        let aligned_prosody =
            self.aligned_prosody(runtime, &token_array, &style, utterance.speed)?;
        let noise_key = match utterance.noise_seed {
            Some(noise_seed) => Some(runtime.random_key(noise_seed)?),
            None => None,
        };
        let decoder_inputs = DecoderInputs {
            aligned_text: &aligned_prosody.aligned_text,
            pitch_curve: &aligned_prosody.pitch_curve,
            energy_curve: &aligned_prosody.energy_curve,
            decoder_style: &style.decoder_style,
            noise_key: noise_key.as_ref(),
        };
        let samples = self.decoder.decode(runtime, &decoder_inputs)?;
        Ok(KokoroForwardPass {
            frame_counts: aligned_prosody.frame_counts,
            pitch_curve: aligned_prosody.pitch_curve,
            energy_curve: aligned_prosody.energy_curve,
            samples,
        })
    }

    fn aligned_prosody(
        &self,
        runtime: &MlxRuntime,
        token_array: &MlxArray,
        style: &UtteranceStyle,
        speed: f32,
    ) -> Result<AlignedProsody, SpeechModelError> {
        let albert_states = self.albert.encode(runtime, token_array)?;
        let durations =
            self.prosody
                .durations(runtime, &albert_states, &style.prosody_style, speed)?;
        let frame_to_phoneme = frame_alignment(runtime, &durations.frame_counts)?;
        let aligned_durations =
            runtime.take_axis(&durations.duration_states, &frame_to_phoneme, 0)?;
        let (pitch_curve, energy_curve) =
            self.prosody
                .pitch_and_energy(runtime, &aligned_durations, &style.prosody_style)?;
        let text_states = self.text_encoder.encode(runtime, token_array)?;
        Ok(AlignedProsody {
            frame_counts: durations.frame_counts,
            pitch_curve,
            energy_curve,
            aligned_text: runtime.take_axis(&text_states, &frame_to_phoneme, 0)?,
        })
    }
}
