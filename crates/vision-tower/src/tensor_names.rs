//! How tensor names reveal what a tensor is, across the naming conventions in
//! the field. Upstream checkpoints say `model.language_model.layers.0...`, MLX
//! conversions say `language_model.model.layers.0...`, and both mean one tensor.

const VISION_TOWER_ROOTS: [&str; 3] = ["vision_tower", "visual", "vision_model"];
const WRAPPER_PREFIXES: [&str; 3] = ["model.", "language_model.", "text_model."];
const BLOCK_CONTAINERS: [&str; 2] = ["blocks", "layers"];
const QUANTIZATION_COMPANION_SUFFIXES: [&str; 2] = [".scales", ".biases"];
/// The suffix a quantized `.weight` always has a sibling with.
pub const QUANTIZATION_SCALES_SUFFIX: &str = ".scales";

/// Whether a tensor belongs to a vision tower, with or without a `model.` wrapper.
#[must_use]
pub fn is_vision_tower_tensor(tensor_name: &str) -> bool {
    let without_model_prefix = tensor_name.strip_prefix("model.").unwrap_or(tensor_name);
    let root_segment = without_model_prefix.split('.').next().unwrap_or_default();
    VISION_TOWER_ROOTS.contains(&root_segment)
}

/// The tower's name prefix as written, such as `vision_tower` or `model.visual`.
#[must_use]
pub fn tower_name_prefix(tensor_name: &str) -> String {
    let mut prefix_segments = Vec::new();
    for name_segment in tensor_name.split('.') {
        prefix_segments.push(name_segment);
        if VISION_TOWER_ROOTS.contains(&name_segment) {
            break;
        }
    }
    prefix_segments.join(".")
}

/// The block index in a name such as `vision_tower.blocks.7.attn.proj.weight`.
#[must_use]
pub fn block_index(tensor_name: &str) -> Option<u64> {
    let mut name_segments = tensor_name.split('.');
    while let Some(name_segment) = name_segments.next() {
        if BLOCK_CONTAINERS.contains(&name_segment) {
            return name_segments.next()?.parse().ok();
        }
    }
    None
}

/// The name with every wrapper removed, so two conventions can be compared.
#[must_use]
pub fn canonical_tensor_name(tensor_name: &str) -> &str {
    let mut remaining_name = tensor_name;
    loop {
        let mut was_stripped = false;
        for wrapper_prefix in WRAPPER_PREFIXES {
            if let Some(unwrapped_name) = remaining_name.strip_prefix(wrapper_prefix) {
                remaining_name = unwrapped_name;
                was_stripped = true;
            }
        }
        if !was_stripped {
            return remaining_name;
        }
    }
}

/// Whether the tensor holds a quantized weight's scales or biases.
#[must_use]
pub fn is_quantization_companion(tensor_name: &str) -> bool {
    for companion_suffix in QUANTIZATION_COMPANION_SUFFIXES {
        if tensor_name.ends_with(companion_suffix) {
            return true;
        }
    }
    false
}

/// The name a weight's scales would have if it were quantized, for asking whether
/// `layer.weight` holds packed codes rather than real values.
#[must_use]
pub fn scales_name_for_weight(tensor_name: &str) -> Option<String> {
    let layer_name = tensor_name.strip_suffix(".weight")?;
    Some(format!("{layer_name}{QUANTIZATION_SCALES_SUFFIX}"))
}

#[cfg(test)]
#[path = "tensor_names_tests.rs"]
mod tests;
