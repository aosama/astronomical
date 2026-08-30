//! Exact JSON configuration profile for the official distilled 4B artifact.

use serde::Deserialize;
use serde::de::DeserializeOwned;
use thiserror::Error;

use super::Flux2KleinOfficialProfile;
use crate::strict_json::DuplicateAwareJsonValue;
const HIDDEN_STATE_TAPS: [usize; 3] = [9, 18, 27];

/// Configuration rejected before any weight descriptor reaches an engine.
#[derive(Debug, Error)]
pub enum Flux2KleinConfigError {
    #[error("malformed FLUX.2 Klein {document} configuration")]
    Malformed {
        document: &'static str,
        #[source]
        source: serde_json::Error,
    },
    #[error("unsupported FLUX.2 Klein profile in {document}: {field}")]
    UnsupportedProfile {
        document: &'static str,
        field: &'static str,
    },
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PipelineDocument {
    #[serde(rename = "_class_name")]
    class_name: String,
    #[serde(rename = "_diffusers_version")]
    diffusers_version: String,
    is_distilled: bool,
    scheduler: [String; 2],
    text_encoder: [String; 2],
    tokenizer: [String; 2],
    transformer: [String; 2],
    vae: [String; 2],
}

/// Validated ownership graph from `model_index.json`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Flux2KleinPipelineConfig {
    is_distilled: bool,
}

impl Flux2KleinPipelineConfig {
    pub fn parse(json_bytes: &[u8]) -> Result<Self, Flux2KleinConfigError> {
        let document: PipelineDocument = parse_document(json_bytes, "model_index.json")?;
        require(
            document.class_name == "Flux2KleinPipeline",
            "model_index.json",
            "_class_name",
        )?;
        require(
            document.diffusers_version == "0.37.0.dev0",
            "model_index.json",
            "_diffusers_version",
        )?;
        require(document.is_distilled, "model_index.json", "is_distilled")?;
        require(
            document.scheduler == ["diffusers", "FlowMatchEulerDiscreteScheduler"],
            "model_index.json",
            "scheduler",
        )?;
        require(
            document.text_encoder == ["transformers", "Qwen3ForCausalLM"],
            "model_index.json",
            "text_encoder",
        )?;
        require(
            document.tokenizer == ["transformers", "Qwen2TokenizerFast"],
            "model_index.json",
            "tokenizer",
        )?;
        require(
            document.transformer == ["diffusers", "Flux2Transformer2DModel"],
            "model_index.json",
            "transformer",
        )?;
        require(
            document.vae == ["diffusers", "AutoencoderKLFlux2"],
            "model_index.json",
            "vae",
        )?;
        Ok(Self { is_distilled: true })
    }

    pub const fn is_distilled(&self) -> bool {
        self.is_distilled
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TransformerDocument {
    #[serde(rename = "_class_name")]
    class_name: String,
    #[serde(rename = "_diffusers_version")]
    diffusers_version: String,
    #[serde(rename = "_name_or_path")]
    name_or_path: String,
    attention_head_dim: usize,
    axes_dims_rope: [usize; 4],
    eps: f64,
    guidance_embeds: bool,
    in_channels: usize,
    joint_attention_dim: usize,
    mlp_ratio: f64,
    num_attention_heads: usize,
    num_layers: usize,
    num_single_layers: usize,
    out_channels: Option<usize>,
    patch_size: usize,
    rope_theta: usize,
    timestep_guidance_channels: usize,
}

/// Geometry consumed directly by the future native transformer owner.
#[derive(Clone, Debug, PartialEq)]
pub struct Flux2KleinTransformerConfig {
    hidden_width: usize,
    attention_head_count: usize,
    attention_head_width: usize,
    input_width: usize,
    conditioning_width: usize,
    feed_forward_width: usize,
    rope_axis_widths: [usize; 4],
    rope_theta: usize,
    double_stream_block_count: usize,
    single_stream_block_count: usize,
    output_width: usize,
    normalization_epsilon: f64,
}

impl Flux2KleinTransformerConfig {
    pub fn parse(json_bytes: &[u8]) -> Result<Self, Flux2KleinConfigError> {
        let document: TransformerDocument = parse_document(json_bytes, "transformer/config.json")?;
        let checks = [
            (
                document.class_name == "Flux2Transformer2DModel",
                "_class_name",
            ),
            (
                document.diffusers_version == "0.37.0.dev0",
                "_diffusers_version",
            ),
            (!document.name_or_path.is_empty(), "_name_or_path"),
            (document.attention_head_dim == 128, "attention_head_dim"),
            (
                document.axes_dims_rope == [32, 32, 32, 32],
                "axes_dims_rope",
            ),
            (document.eps == 0.000_001, "eps"),
            (!document.guidance_embeds, "guidance_embeds"),
            (document.in_channels == 128, "in_channels"),
            (document.joint_attention_dim == 7_680, "joint_attention_dim"),
            (document.mlp_ratio == 3.0, "mlp_ratio"),
            (document.num_attention_heads == 24, "num_attention_heads"),
            (document.num_layers == 5, "num_layers"),
            (document.num_single_layers == 20, "num_single_layers"),
            (document.out_channels.is_none(), "out_channels"),
            (document.patch_size == 1, "patch_size"),
            (document.rope_theta == 2_000, "rope_theta"),
            (
                document.timestep_guidance_channels == 256,
                "timestep_guidance_channels",
            ),
        ];
        require_checks("transformer/config.json", &checks)?;
        let hidden_width = document.attention_head_dim * document.num_attention_heads;
        Ok(Self {
            hidden_width,
            attention_head_count: document.num_attention_heads,
            attention_head_width: document.attention_head_dim,
            input_width: document.in_channels,
            conditioning_width: document.joint_attention_dim,
            feed_forward_width: hidden_width * 3,
            rope_axis_widths: document.axes_dims_rope,
            rope_theta: document.rope_theta,
            double_stream_block_count: document.num_layers,
            single_stream_block_count: document.num_single_layers,
            output_width: document.out_channels.unwrap_or(document.in_channels),
            normalization_epsilon: document.eps,
        })
    }

    pub const fn hidden_width(&self) -> usize {
        self.hidden_width
    }
    pub const fn feed_forward_width(&self) -> usize {
        self.feed_forward_width
    }
    pub const fn attention_head_count(&self) -> usize {
        self.attention_head_count
    }
    pub const fn attention_head_width(&self) -> usize {
        self.attention_head_width
    }
    pub const fn input_width(&self) -> usize {
        self.input_width
    }
    pub const fn conditioning_width(&self) -> usize {
        self.conditioning_width
    }
    pub const fn rope_axis_widths(&self) -> [usize; 4] {
        self.rope_axis_widths
    }
    pub const fn rope_theta(&self) -> usize {
        self.rope_theta
    }
    pub const fn double_stream_block_count(&self) -> usize {
        self.double_stream_block_count
    }
    pub const fn single_stream_block_count(&self) -> usize {
        self.single_stream_block_count
    }
    pub const fn output_width(&self) -> usize {
        self.output_width
    }
    pub const fn normalization_epsilon(&self) -> f64 {
        self.normalization_epsilon
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TextEncoderDocument {
    architectures: [String; 1],
    attention_bias: bool,
    attention_dropout: f64,
    bos_token_id: u32,
    dtype: String,
    eos_token_id: u32,
    head_dim: usize,
    hidden_act: String,
    hidden_size: usize,
    initializer_range: f64,
    intermediate_size: usize,
    layer_types: Vec<String>,
    max_position_embeddings: usize,
    max_window_layers: usize,
    model_type: String,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    num_key_value_heads: usize,
    rms_norm_eps: f64,
    rope_scaling: Option<serde_json::Value>,
    rope_theta: usize,
    sliding_window: Option<usize>,
    tie_word_embeddings: bool,
    transformers_version: String,
    use_cache: bool,
    use_sliding_window: bool,
    vocab_size: usize,
}

/// Strict Qwen3-4B conditioning contract with retained hidden-state tap semantics.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Flux2KleinTextEncoderConfig {
    conditioning_width: usize,
}

impl Flux2KleinTextEncoderConfig {
    pub fn parse(json_bytes: &[u8]) -> Result<Self, Flux2KleinConfigError> {
        let document: TextEncoderDocument = parse_document(json_bytes, "text_encoder/config.json")?;
        let checks = [
            (
                document.architectures == ["Qwen3ForCausalLM"],
                "architectures",
            ),
            (!document.attention_bias, "attention_bias"),
            (document.attention_dropout == 0.0, "attention_dropout"),
            (document.bos_token_id == 151_643, "bos_token_id"),
            (document.dtype == "bfloat16", "dtype"),
            (document.eos_token_id == 151_645, "eos_token_id"),
            (document.head_dim == 128, "head_dim"),
            (document.hidden_act == "silu", "hidden_act"),
            (document.hidden_size == 2_560, "hidden_size"),
            (document.initializer_range == 0.02, "initializer_range"),
            (document.intermediate_size == 9_728, "intermediate_size"),
            (
                document.layer_types.len() == 36
                    && document
                        .layer_types
                        .iter()
                        .all(|kind| kind == "full_attention"),
                "layer_types",
            ),
            (
                document.max_position_embeddings == 40_960,
                "max_position_embeddings",
            ),
            (document.max_window_layers == 36, "max_window_layers"),
            (document.model_type == "qwen3", "model_type"),
            (document.num_attention_heads == 32, "num_attention_heads"),
            (document.num_hidden_layers == 36, "num_hidden_layers"),
            (document.num_key_value_heads == 8, "num_key_value_heads"),
            (document.rms_norm_eps == 0.000_001, "rms_norm_eps"),
            (document.rope_scaling.is_none(), "rope_scaling"),
            (document.rope_theta == 1_000_000, "rope_theta"),
            (document.sliding_window.is_none(), "sliding_window"),
            (document.tie_word_embeddings, "tie_word_embeddings"),
            (
                document.transformers_version == "4.56.1",
                "transformers_version",
            ),
            (document.use_cache, "use_cache"),
            (!document.use_sliding_window, "use_sliding_window"),
            (document.vocab_size == 151_936, "vocab_size"),
        ];
        require_checks("text_encoder/config.json", &checks)?;
        Ok(Self {
            conditioning_width: document.hidden_size * HIDDEN_STATE_TAPS.len(),
        })
    }

    pub const fn hidden_state_taps(&self) -> &[usize; 3] {
        &HIDDEN_STATE_TAPS
    }
    pub const fn conditioning_width(&self) -> usize {
        self.conditioning_width
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct VaeDocument {
    #[serde(rename = "_class_name")]
    class_name: String,
    #[serde(rename = "_diffusers_version")]
    diffusers_version: String,
    #[serde(rename = "_name_or_path")]
    name_or_path: String,
    act_fn: String,
    batch_norm_eps: f64,
    batch_norm_momentum: f64,
    block_out_channels: [usize; 4],
    down_block_types: [String; 4],
    force_upcast: bool,
    in_channels: usize,
    latent_channels: usize,
    layers_per_block: usize,
    mid_block_add_attention: bool,
    norm_num_groups: usize,
    out_channels: usize,
    patch_size: [usize; 2],
    sample_size: usize,
    up_block_types: [String; 4],
    use_post_quant_conv: bool,
    use_quant_conv: bool,
}

/// Exact VAE topology needed by complete decoding.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Flux2KleinVaeConfig {
    latent_channel_count: usize,
}

impl Flux2KleinVaeConfig {
    pub fn parse(json_bytes: &[u8]) -> Result<Self, Flux2KleinConfigError> {
        let document: VaeDocument = parse_document(json_bytes, "vae/config.json")?;
        let expected_down = ["DownEncoderBlock2D"; 4];
        let expected_up = ["UpDecoderBlock2D"; 4];
        let checks = [
            (document.class_name == "AutoencoderKLFlux2", "_class_name"),
            (
                document.diffusers_version == "0.37.0.dev0",
                "_diffusers_version",
            ),
            (!document.name_or_path.is_empty(), "_name_or_path"),
            (document.act_fn == "silu", "act_fn"),
            (document.batch_norm_eps == 0.000_1, "batch_norm_eps"),
            (document.batch_norm_momentum == 0.1, "batch_norm_momentum"),
            (
                document.block_out_channels == [128, 256, 512, 512],
                "block_out_channels",
            ),
            (
                document.down_block_types == expected_down,
                "down_block_types",
            ),
            (document.force_upcast, "force_upcast"),
            (document.in_channels == 3, "in_channels"),
            (document.latent_channels == 32, "latent_channels"),
            (document.layers_per_block == 2, "layers_per_block"),
            (document.mid_block_add_attention, "mid_block_add_attention"),
            (document.norm_num_groups == 32, "norm_num_groups"),
            (document.out_channels == 3, "out_channels"),
            (document.patch_size == [2, 2], "patch_size"),
            (document.sample_size == 1_024, "sample_size"),
            (document.up_block_types == expected_up, "up_block_types"),
            (document.use_post_quant_conv, "use_post_quant_conv"),
            (document.use_quant_conv, "use_quant_conv"),
        ];
        require_checks("vae/config.json", &checks)?;
        Ok(Self {
            latent_channel_count: document.latent_channels,
        })
    }
    pub const fn latent_channel_count(&self) -> usize {
        self.latent_channel_count
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SchedulerDocument {
    #[serde(rename = "_class_name")]
    class_name: String,
    #[serde(rename = "_diffusers_version")]
    diffusers_version: String,
    base_image_seq_len: usize,
    base_shift: f64,
    invert_sigmas: bool,
    max_image_seq_len: usize,
    max_shift: f64,
    num_train_timesteps: usize,
    shift: f64,
    shift_terminal: Option<f64>,
    stochastic_sampling: bool,
    time_shift_type: String,
    use_beta_sigmas: bool,
    use_dynamic_shifting: bool,
    use_exponential_sigmas: bool,
    use_karras_sigmas: bool,
}

/// CPU-scalar scheduler constants matched to the artifact configuration.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Flux2KleinSchedulerConfig;

impl Flux2KleinSchedulerConfig {
    pub fn parse(json_bytes: &[u8]) -> Result<Self, Flux2KleinConfigError> {
        let document: SchedulerDocument =
            parse_document(json_bytes, "scheduler/scheduler_config.json")?;
        let checks = [
            (
                document.class_name == "FlowMatchEulerDiscreteScheduler",
                "_class_name",
            ),
            (
                document.diffusers_version == "0.37.0.dev0",
                "_diffusers_version",
            ),
            (document.base_image_seq_len == 256, "base_image_seq_len"),
            (document.base_shift == 0.5, "base_shift"),
            (!document.invert_sigmas, "invert_sigmas"),
            (document.max_image_seq_len == 4_096, "max_image_seq_len"),
            (document.max_shift == 1.15, "max_shift"),
            (document.num_train_timesteps == 1_000, "num_train_timesteps"),
            (document.shift == 3.0, "shift"),
            (document.shift_terminal.is_none(), "shift_terminal"),
            (!document.stochastic_sampling, "stochastic_sampling"),
            (document.time_shift_type == "exponential", "time_shift_type"),
            (!document.use_beta_sigmas, "use_beta_sigmas"),
            (document.use_dynamic_shifting, "use_dynamic_shifting"),
            (!document.use_exponential_sigmas, "use_exponential_sigmas"),
            (!document.use_karras_sigmas, "use_karras_sigmas"),
        ];
        require_checks("scheduler/scheduler_config.json", &checks)?;
        Ok(Self)
    }
    pub const fn inference_step_count(&self) -> usize {
        Flux2KleinOfficialProfile::inference_step_count()
    }
}

fn parse_document<T: DeserializeOwned>(
    json_bytes: &[u8],
    document: &'static str,
) -> Result<T, Flux2KleinConfigError> {
    let duplicate_aware = serde_json::from_slice::<DuplicateAwareJsonValue>(json_bytes)
        .map_err(|source| Flux2KleinConfigError::Malformed { document, source })?;
    serde_json::from_value(duplicate_aware.0)
        .map_err(|source| Flux2KleinConfigError::Malformed { document, source })
}

fn require(
    is_supported: bool,
    document: &'static str,
    field: &'static str,
) -> Result<(), Flux2KleinConfigError> {
    if is_supported {
        Ok(())
    } else {
        Err(Flux2KleinConfigError::UnsupportedProfile { document, field })
    }
}

fn require_checks(
    document: &'static str,
    checks: &[(bool, &'static str)],
) -> Result<(), Flux2KleinConfigError> {
    for (is_supported, field) in checks {
        require(*is_supported, document, field)?;
    }
    Ok(())
}
