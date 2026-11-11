//! The parts of Phonon's `config.json` the forward pass depends on.
//!
//! Phonon descends from Kyutai's Pocket TTS, whose checkpoints vary in width,
//! depth and sampler. This port reproduces the published Phonon 31M English
//! checkpoint; reading its shape from the file and refusing values the port
//! was not written for turns a different checkpoint into a clear load error
//! rather than noise.

use serde_json::Value;

use crate::speech_models::speech_model_error::SpeechModelError;

/// The flow language model's shape.
#[derive(Clone, Debug)]
pub(crate) struct FlowModelShape {
    pub(crate) head_count: i32,
    pub(crate) layer_count: usize,
    pub(crate) latent_width: i32,
    pub(crate) flow_depth: usize,
    pub(crate) rotary_max_period: f32,
}

/// The Mimi decoder's shape.
#[derive(Clone, Debug)]
pub(crate) struct MimiDecoderShape {
    pub(crate) head_count: i32,
    pub(crate) layer_count: usize,
    /// How many past positions (itself included) each position attends to.
    pub(crate) attention_context: usize,
    /// Whether each transformer branch is scaled per channel (layer scale).
    pub(crate) has_layer_scale: bool,
    pub(crate) rotary_max_period: f32,
    /// SEANet upsampling ratios, applied in this order.
    pub(crate) upsampling_ratios: Vec<i32>,
    pub(crate) sample_rate_hertz: u32,
    pub(crate) frame_rate_hertz: f64,
}

#[derive(Clone, Debug)]
pub(crate) struct PhononConfiguration {
    pub(crate) flow: FlowModelShape,
    pub(crate) mimi: MimiDecoderShape,
    /// End of speech when the end-of-speech logit exceeds this.
    pub(crate) end_of_speech_threshold: f32,
    /// The `possible_values` of the speaker-count conditioner.
    pub(crate) speaker_count_values: Vec<String>,
    pub(crate) duration_conditioner: DurationConditioner,
}

/// The continuous duration conditioner: a sinusoid of `scale_factor * delta`.
#[derive(Clone, Debug)]
pub(crate) struct DurationConditioner {
    pub(crate) scale_factor: f32,
    pub(crate) width: usize,
    pub(crate) max_period: f32,
}

impl PhononConfiguration {
    pub(crate) fn parse(configuration_json: &str) -> Result<Self, SpeechModelError> {
        let document: Value = serde_json::from_str(configuration_json).map_err(|parse_error| {
            SpeechModelError::Artifact(format!("config.json is not JSON: {parse_error}"))
        })?;
        let flow_section = &document["flow_lm"];
        require_text(flow_section, "time_rms_norm", "rms")?;
        require_text(flow_section, "gelu", "erf")?;
        require_unsigned(&document, "lsd_decode_steps", 1)?;
        if document["prefix_sinks"].is_object() {
            return Err(SpeechModelError::Artifact(
                "checkpoints with prefix sinks are not supported".to_owned(),
            ));
        }
        require_fuser(&document["fuser"])?;
        Ok(Self {
            flow: flow_shape(flow_section)?,
            mimi: mimi_shape(&document["mimi"])?,
            end_of_speech_threshold: number(&document, "eos_threshold")? as f32,
            speaker_count_values: speaker_count_values(&document)?,
            duration_conditioner: duration_conditioner(&document)?,
        })
    }

    pub(crate) fn sample_rate_hertz(&self) -> u32 {
        self.mimi.sample_rate_hertz
    }

    /// Audio samples per generated frame (80 ms at 24 kHz).
    pub(crate) fn samples_per_frame(&self) -> usize {
        (f64::from(self.mimi.sample_rate_hertz) / self.mimi.frame_rate_hertz).round() as usize
    }
}

fn flow_shape(flow_section: &Value) -> Result<FlowModelShape, SpeechModelError> {
    Ok(FlowModelShape {
        head_count: positive_integer(flow_section, "num_heads")?,
        layer_count: positive_integer(flow_section, "num_layers")? as usize,
        latent_width: positive_integer(flow_section, "ldim")?,
        flow_depth: positive_integer(flow_section, "flow_depth")? as usize,
        rotary_max_period: number(flow_section, "max_period")? as f32,
    })
}

fn mimi_shape(mimi_section: &Value) -> Result<MimiDecoderShape, SpeechModelError> {
    let upsampling_ratios = mimi_section["ratios"]
        .as_array()
        .ok_or_else(|| SpeechModelError::Artifact("config.json mimi has no ratios".to_owned()))?
        .iter()
        .map(|ratio| {
            ratio
                .as_i64()
                .filter(|ratio| *ratio > 0)
                .map(|ratio| ratio as i32)
        })
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| {
            SpeechModelError::Artifact(
                "config.json mimi ratios are not positive integers".to_owned(),
            )
        })?;
    require_unsigned(mimi_section, "n_residual_layers", 1)?;
    require_unsigned(mimi_section, "compress", 2)?;
    Ok(MimiDecoderShape {
        head_count: positive_integer(mimi_section, "transformer_num_heads")?,
        layer_count: positive_integer(mimi_section, "transformer_num_layers")? as usize,
        attention_context: positive_integer(mimi_section, "transformer_context")? as usize,
        has_layer_scale: mimi_section["transformer_layer_scale"].as_f64().is_some(),
        rotary_max_period: number(mimi_section, "transformer_max_period")? as f32,
        upsampling_ratios,
        sample_rate_hertz: positive_integer(mimi_section, "sample_rate")? as u32,
        frame_rate_hertz: positive_number(mimi_section, "frame_rate")?,
    })
}

/// The port sums the speaker-count and duration conditions into each frame
/// and prompts with the voice and the text; any other fusion is refused.
fn require_fuser(fuser: &Value) -> Result<(), SpeechModelError> {
    let summed_conditions = text_list(&fuser["sum"]);
    let prepended_prompts = text_list(&fuser["prepend"]);
    let is_supported_fusion = summed_conditions == ["num_speakers", "duration_delta"]
        && prepended_prompts
            .iter()
            .all(|prompt| prompt == "transcript_in_segment" || prompt == "speaker_wavs")
        && text_list(&fuser["cross"]).is_empty();
    if is_supported_fusion {
        return Ok(());
    }
    Err(SpeechModelError::Artifact(format!(
        "config.json fuser sums {summed_conditions:?} and prepends {prepended_prompts:?}; this engine \
         reproduces Phonon 31M, which sums num_speakers and duration_delta"
    )))
}

fn text_list(list: &Value) -> Vec<String> {
    list.as_array()
        .map(|entries| {
            entries
                .iter()
                .filter_map(|entry| entry.as_str().map(str::to_owned))
                .collect()
        })
        .unwrap_or_default()
}

fn duration_conditioner(document: &Value) -> Result<DurationConditioner, SpeechModelError> {
    let conditioners = document["conditioners"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    let duration_section = conditioners
        .iter()
        .find(|conditioner| conditioner["name"] == "duration_delta")
        .map(|conditioner| &conditioner["continuous"])
        .ok_or_else(|| {
            SpeechModelError::Artifact("config.json has no duration_delta conditioner".to_owned())
        })?;
    Ok(DurationConditioner {
        scale_factor: number(duration_section, "scale_factor")? as f32,
        width: positive_integer(duration_section, "dim")? as usize,
        max_period: number(duration_section, "max_period")? as f32,
    })
}

fn speaker_count_values(document: &Value) -> Result<Vec<String>, SpeechModelError> {
    let conditioners = document["conditioners"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    let speaker_count = conditioners
        .iter()
        .find(|conditioner| conditioner["name"] == "num_speakers");
    let Some(speaker_count) = speaker_count else {
        return Ok(Vec::new());
    };
    speaker_count["lut"]["possible_values"]
        .as_array()
        .and_then(|values| {
            values
                .iter()
                .map(|value| value.as_str().map(str::to_owned))
                .collect()
        })
        .ok_or_else(|| SpeechModelError::Artifact("num_speakers has no possible_values".to_owned()))
}

/// Every count and width the forward pass divides by, loops over or sizes a
/// window with must be at least 1; zero or a negative number is refused here
/// rather than hanging or dividing by zero later.
fn positive_integer(section: &Value, key: &str) -> Result<i32, SpeechModelError> {
    section[key]
        .as_i64()
        .filter(|found| *found > 0)
        .and_then(|found| i32::try_from(found).ok())
        .ok_or_else(|| {
            SpeechModelError::Artifact(format!("config.json has no positive integer {key}"))
        })
}

/// A rate the frame budget and samples per frame are divided from; zero or
/// a negative rate would give an empty budget and no audio at all.
fn positive_number(section: &Value, key: &str) -> Result<f64, SpeechModelError> {
    section[key]
        .as_f64()
        .filter(|found| found.is_finite() && *found > 0.0)
        .ok_or_else(|| {
            SpeechModelError::Artifact(format!("config.json has no positive number {key}"))
        })
}

fn number(section: &Value, key: &str) -> Result<f64, SpeechModelError> {
    section[key]
        .as_f64()
        .ok_or_else(|| SpeechModelError::Artifact(format!("config.json has no number {key}")))
}

fn require_unsigned(section: &Value, key: &str, expected: u64) -> Result<(), SpeechModelError> {
    match section[key].as_u64() {
        Some(found) if found == expected => Ok(()),
        found => Err(SpeechModelError::Artifact(format!(
            "config.json {key} is {found:?}; this engine reproduces Phonon 31M, where it is {expected}"
        ))),
    }
}

fn require_text(section: &Value, key: &str, expected: &str) -> Result<(), SpeechModelError> {
    match section[key].as_str() {
        Some(found) if found == expected => Ok(()),
        found => Err(SpeechModelError::Artifact(format!(
            "config.json {key} is {found:?}; this engine reproduces Phonon 31M, where it is {expected:?}"
        ))),
    }
}

#[cfg(test)]
#[path = "configuration_tests.rs"]
mod tests;
