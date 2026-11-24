//! Where each source tensor lands in the converted artifact, and in what
//! spelling and axis order.
//!
//! OptiQ never writes a source checkpoint's own names: it converts through
//! mlx-lm, whose model class `sanitize` renames the language tower, drops
//! the vision tower and the multi-token-prediction head, moves convolution
//! axes into MLX order, and materializes RMSNorm gammas. OptiQ then puts the
//! towers back as a bfloat16 sidecar. A converter that walks shards instead
//! of instantiating the model has to apply the same rules itself, or an
//! original Hugging Face checkpoint comes out under names no OptiQ-format
//! reader looks up. This module is those rules as data, decided from names
//! and header shapes alone.
//!
//! Ported from mlx-lm's `qwen3_5.Model.sanitize`, `qwen3_5.TextModel.sanitize`
//! and `qwen3_5_moe.Model.sanitize`, and from OptiQ's `_is_mm_key`,
//! `_store_key` and `_sanitize_conv` (`optiq/vlm/sidecar.py`).

use std::collections::BTreeMap;

#[cfg(test)]
#[path = "output_layout_tests.rs"]
mod output_layout_tests;

/// The file a tensor is written to.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum OutputDestination {
    /// The `model*.safetensors` shards every loader reads.
    ModelShards,
    /// `optiq_vision.safetensors`, the bfloat16 towers beside the shards.
    VisionSidecar,
    /// `optiq/mtp.safetensors`, the multi-token-prediction head, which no
    /// model class has a module for.
    MtpSidecar,
    /// Not written: a head tied to the embedding, or a tower the config
    /// does not declare.
    Dropped,
}

/// What happens to a tensor's values on the way out.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum OutputValueTransform {
    Unchanged,
    /// Reorder the axes, PyTorch convolution layout to MLX layout.
    PermuteAxes(Vec<i32>),
    /// Original releases store an RMSNorm weight as `gamma - 1`.
    AddOne,
}

/// One source tensor's place in the output.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OutputTensorLayout {
    pub output_name: String,
    pub destination: OutputDestination,
    pub value_transform: OutputValueTransform,
    /// mlx-lm's `cast_predicate` exempts the GatedDeltaNet `A_log` from the
    /// bfloat16 cast: its exponent underflows at reduced precision.
    pub keeps_stored_dtype: bool,
}

/// The layout rules for one source directory.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OutputLayoutRules {
    applies_qwen3_5_sanitize: bool,
    has_unsanitized_convolution: bool,
    ties_word_embeddings: bool,
    declares_multimodal_towers: bool,
}

/// Weight-name prefixes of vision and audio towers and their projectors,
/// OptiQ's `_MM_PREFIXES`, matched after a leading `model.` (and then
/// `encoder.`) is stripped.
const MULTIMODAL_PREFIXES: [&str; 11] = [
    "vision_tower.",
    "audio_tower.",
    "multi_modal_projector.",
    "embed_vision.",
    "embed_audio.",
    "vision_embedder.",
    "audio_embedder.",
    "visual.",
    "vision_adapter.",
    "vision_projection.",
    "merger.",
];

/// The 1-D RMSNorm weights mlx-lm offsets by one for an original release.
const OFFSET_NORM_SUFFIXES: [&str; 5] = [
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    "model.norm.weight",
    ".q_norm.weight",
    ".k_norm.weight",
];

impl OutputLayoutRules {
    /// Rules that leave every name and value as stored and keep towers in
    /// the model shards: the behavior for a source without a `config.json`.
    #[must_use]
    pub const fn unchanged() -> Self {
        Self {
            applies_qwen3_5_sanitize: false,
            has_unsanitized_convolution: false,
            ties_word_embeddings: false,
            declares_multimodal_towers: false,
        }
    }

    /// Reads the family, the tied-embedding flag and the tower declaration
    /// from the model config, and the convolution layout from the header
    /// shapes. mlx-lm keys the norm offset off the convolution layout, so a
    /// checkpoint already in MLX layout is never offset twice.
    #[must_use]
    pub fn from_config(
        config_value: &serde_json::Value,
        shape_by_tensor_name: &BTreeMap<String, Vec<usize>>,
    ) -> Self {
        let model_type = config_value["model_type"].as_str().unwrap_or_default();
        let text_model_type = config_value["text_config"]["model_type"]
            .as_str()
            .unwrap_or_default();
        let applies_qwen3_5_sanitize =
            model_type.starts_with("qwen3_5") || text_model_type.starts_with("qwen3_5");
        let text_config = if config_value["text_config"].is_object() {
            &config_value["text_config"]
        } else {
            config_value
        };
        let mut has_unsanitized_convolution = false;
        for (tensor_name, shape) in shape_by_tensor_name {
            if tensor_name.contains("conv1d.weight") && shape.last().copied() != Some(1) {
                has_unsanitized_convolution = true;
            }
        }
        Self {
            applies_qwen3_5_sanitize,
            has_unsanitized_convolution,
            ties_word_embeddings: text_config["tie_word_embeddings"]
                .as_bool()
                .unwrap_or(false),
            declares_multimodal_towers: config_value["vision_config"].is_object()
                || config_value["audio_config"].is_object(),
        }
    }

    /// Whether the config declares a vision or audio tower.
    #[must_use]
    pub const fn declares_multimodal_towers(&self) -> bool {
        self.declares_multimodal_towers
    }

    /// Whether the family's mixture-of-experts router is a plain linear
    /// layer that mlx-lm quantizes like any other (Qwen's `mlp.gate`).
    #[must_use]
    pub const fn quantizes_router_linears(&self) -> bool {
        self.applies_qwen3_5_sanitize
    }

    /// The output name of a module whose source-spelled name is
    /// `module_name` (no `.weight`). Renames are prefix rewrites, so they
    /// compose with the expert renames the inventory applies first.
    #[must_use]
    pub fn output_module_name(&self, module_name: &str) -> String {
        if self.applies_qwen3_5_sanitize {
            return qwen3_5_language_name(module_name);
        }
        module_name.to_owned()
    }

    /// The layout of one source tensor with header `shape`.
    #[must_use]
    pub fn layout_for(&self, tensor_name: &str, shape: &[usize]) -> OutputTensorLayout {
        if is_multimodal_tensor(tensor_name) {
            return self.tower_layout(tensor_name, shape);
        }
        if !self.applies_qwen3_5_sanitize {
            return unchanged_layout(tensor_name);
        }
        // The head keeps the release's own names: its loader reads them, and
        // mlx-lm's rename is for tensors it has modules for.
        if tensor_name.starts_with("mtp.") {
            return OutputTensorLayout {
                output_name: tensor_name.to_owned(),
                destination: OutputDestination::MtpSidecar,
                value_transform: OutputValueTransform::Unchanged,
                keeps_stored_dtype: true,
            };
        }
        let output_name = qwen3_5_language_name(tensor_name);
        let is_tied_head = self.ties_word_embeddings && output_name.ends_with("lm_head.weight");
        if output_name.contains("mtp.") || is_tied_head {
            return OutputTensorLayout {
                output_name,
                destination: OutputDestination::Dropped,
                value_transform: OutputValueTransform::Unchanged,
                keeps_stored_dtype: false,
            };
        }
        let value_transform = self.language_value_transform(&output_name, shape);
        OutputTensorLayout {
            keeps_stored_dtype: output_name.ends_with("A_log"),
            output_name,
            destination: OutputDestination::ModelShards,
            value_transform,
        }
    }

    /// A tower tensor goes to the sidecar under OptiQ's stored name when the
    /// config declares a tower. Undeclared towers are dropped for a family
    /// whose mlx-lm class drops them, and otherwise stay where they were.
    fn tower_layout(&self, tensor_name: &str, shape: &[usize]) -> OutputTensorLayout {
        if !self.declares_multimodal_towers {
            let mut layout = unchanged_layout(tensor_name);
            if self.applies_qwen3_5_sanitize {
                layout.destination = OutputDestination::Dropped;
            }
            return layout;
        }
        let output_name = sidecar_store_name(tensor_name);
        OutputTensorLayout {
            value_transform: sidecar_convolution_transform(&output_name, shape),
            output_name,
            destination: OutputDestination::VisionSidecar,
            keeps_stored_dtype: false,
        }
    }

    /// mlx-lm's `TextModel.sanitize`: the depthwise convolution moves from
    /// `[channels, 1, kernel]` to `[channels, kernel, 1]`, and when any
    /// convolution was in the PyTorch layout the 1-D norm weights gain one.
    fn language_value_transform(&self, output_name: &str, shape: &[usize]) -> OutputValueTransform {
        let is_unsanitized_convolution = output_name.contains("conv1d.weight")
            && shape.len() == 3
            && shape.last().copied() != Some(1);
        if is_unsanitized_convolution {
            return OutputValueTransform::PermuteAxes(vec![0, 2, 1]);
        }
        let is_offset_norm = self.has_unsanitized_convolution
            && shape.len() == 1
            && has_any_suffix(output_name, &OFFSET_NORM_SUFFIXES);
        if is_offset_norm {
            return OutputValueTransform::AddOne;
        }
        OutputValueTransform::Unchanged
    }
}

/// OptiQ's `_is_mm_key`.
#[must_use]
pub fn is_multimodal_tensor(tensor_name: &str) -> bool {
    let container_stripped_name = without_tower_containers(tensor_name);
    for multimodal_prefix in MULTIMODAL_PREFIXES {
        if container_stripped_name.starts_with(multimodal_prefix) {
            return true;
        }
    }
    false
}

/// The base's containers a tower sits inside: `model.`, then DiffusionGemma's
/// `encoder.`.
fn without_tower_containers(tensor_name: &str) -> &str {
    let without_model = tensor_name.strip_prefix("model.").unwrap_or(tensor_name);
    without_model
        .strip_prefix("encoder.")
        .unwrap_or(without_model)
}

/// OptiQ's `_store_key`: containers stripped, and the Qwen `visual.` tower
/// renamed to the `vision_tower.` the front-ends load.
fn sidecar_store_name(tensor_name: &str) -> String {
    let container_stripped_name = without_tower_containers(tensor_name);
    match container_stripped_name.strip_prefix("visual.") {
        Some(tower_suffix) => format!("vision_tower.{tower_suffix}"),
        None => container_stripped_name.to_owned(),
    }
}

/// OptiQ's `_sanitize_conv`: PyTorch channel-first convolution weights to
/// MLX channel-last, each guarded by the shape only the PyTorch layout has
/// so an already-converted tower is left alone.
fn sidecar_convolution_transform(store_name: &str, shape: &[usize]) -> OutputValueTransform {
    if is_channel_first_2d_convolution(store_name, shape) {
        return OutputValueTransform::PermuteAxes(vec![0, 2, 3, 1]);
    }
    if store_name.contains("depthwise_conv1d.weight") && shape.len() == 3 && shape[1] == 1 {
        return OutputValueTransform::PermuteAxes(vec![0, 2, 1]);
    }
    if store_name.contains("patch_embed.proj.weight") && shape.len() == 5 && shape[4] != 3 {
        return OutputValueTransform::PermuteAxes(vec![0, 2, 3, 4, 1]);
    }
    OutputValueTransform::Unchanged
}

/// The three 2-D convolutions OptiQ moves to channel-last, each recognized
/// by the square kernel only the PyTorch layout has in its trailing pair:
/// Gemma-4's audio subsampling, LFM3's depthwise spatial convolution (one
/// input channel), and a Mistral or Pixtral patch embedding.
fn is_channel_first_2d_convolution(store_name: &str, shape: &[usize]) -> bool {
    if shape.len() != 4 || shape[2] != shape[3] {
        return false;
    }
    let is_audio_subsampling =
        store_name.contains("subsample_conv_projection") && store_name.contains("conv.weight");
    let is_spatial_convolution = store_name.contains("spatial_conv.weight") && shape[1] == 1;
    let is_patch_convolution = store_name.contains("patch_conv.weight") && shape[1] != shape[2];
    is_audio_subsampling || is_spatial_convolution || is_patch_convolution
}

/// mlx-lm's Qwen3.5 wrapper rename: `model.language_model*` becomes
/// `language_model.model*`, a name already under `language_model.` stays,
/// and anything else moves under `language_model.`.
fn qwen3_5_language_name(source_name: &str) -> String {
    if let Some(tower_suffix) = source_name.strip_prefix("model.language_model") {
        return format!("language_model.model{tower_suffix}");
    }
    if source_name.starts_with("language_model.") {
        return source_name.to_owned();
    }
    format!("language_model.{source_name}")
}

fn unchanged_layout(tensor_name: &str) -> OutputTensorLayout {
    OutputTensorLayout {
        output_name: tensor_name.to_owned(),
        destination: OutputDestination::ModelShards,
        value_transform: OutputValueTransform::Unchanged,
        keeps_stored_dtype: false,
    }
}

fn has_any_suffix(tensor_name: &str, suffixes: &[&str]) -> bool {
    for suffix in suffixes {
        if tensor_name.ends_with(suffix) {
            return true;
        }
    }
    false
}
