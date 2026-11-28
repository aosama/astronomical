//! The two folder layouts a Kev decision model is served from.
//!
//! Kev ships as a LoRA adapter and a pickled pointer head beside a separate
//! Qwen3.5 base, which no engine serves as it is. Two merged layouts exist:
//!
//! - **Converted**: what `cargo xtask convert-kev` writes. The settings are in
//!   `kev_decision.json` (`model_type`, `temperature`, `pointer_dimension`,
//!   `base_model`) and the head in `kev_head.safetensors`.
//! - **Published**: the merged MLX builds published on Hugging Face as
//!   `AbstractFramework/kev-*-mlx-*`, which the download catalog installs as
//!   they are. The settings are in `decision.json` (`kind`, `temperature`,
//!   `head_dim`, `base`, `head_file`, `backbone`) and the head in
//!   `head.safetensors`, with the same `q` and `k` tensors in float32.
//!
//! Both describe the same model: on the 26-question float32 reference, the
//! published 8-bit Kev-0.8B gave the same 26 answers with a largest
//! probability difference of 0.021 (the bfloat16 conversion: 0.012).
//! Parsing lives here, apart from any filesystem access, so discovery, the
//! engine and training read a folder the same way.

/// Which of the two layouts a Kev folder uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KevFolderLayout {
    Converted,
    Published,
}

/// What every reader needs from a Kev folder's settings.
#[derive(Clone, Debug, PartialEq)]
pub struct KevFolderSettings {
    /// What every logit is divided by before the softmax.
    pub temperature: f64,
    /// The width of the head's query and key projections.
    pub pointer_dimension: usize,
    /// The Qwen3.5 base the adapter was trained on, such as
    /// `Qwen/Qwen3.5-0.8B-Base`.
    pub base_model: Option<String>,
}

/// The `format` prefix a published `decision.json` declares.
const PUBLISHED_FORMAT_PREFIX: &str = "abstractcore-decision/";

impl KevFolderLayout {
    /// The layouts in the order a folder is checked: a folder carrying both
    /// files is read as converted, the layout Astronomical writes itself.
    pub const ALL: [Self; 2] = [Self::Converted, Self::Published];

    /// The settings file that marks a folder of this layout as Kev.
    #[must_use]
    pub const fn settings_file_name(self) -> &'static str {
        match self {
            Self::Converted => "kev_decision.json",
            Self::Published => "decision.json",
        }
    }

    /// The pointer head's safetensors file.
    #[must_use]
    pub const fn head_file_name(self) -> &'static str {
        match self {
            Self::Converted => "kev_head.safetensors",
            Self::Published => "head.safetensors",
        }
    }

    /// Whether `settings_text` marks a Kev folder of this layout, without
    /// checking the values a reader needs.
    #[must_use]
    pub fn marks_kev(self, settings_text: &[u8]) -> bool {
        let Ok(settings) = serde_json::from_slice::<serde_json::Value>(settings_text) else {
            return false;
        };
        match self {
            Self::Converted => text_field(&settings, "model_type") == Some("kev"),
            Self::Published => {
                text_field(&settings, "kind") == Some("kev")
                    && is_published_format(text_field(&settings, "format"))
            }
        }
    }

    /// The settings `settings_text` holds for this layout.
    ///
    /// # Errors
    ///
    /// Returns why the settings cannot be served: not JSON, not Kev, a
    /// temperature or head width that is not positive, or, for the published
    /// layout, a head file other than `head.safetensors` or a backbone the
    /// adapter is not merged into.
    pub fn read_settings(self, settings_text: &[u8]) -> Result<KevFolderSettings, String> {
        if !self.marks_kev(settings_text) {
            return Err("it does not describe a Kev model".to_owned());
        }
        let settings: serde_json::Value =
            serde_json::from_slice(settings_text).map_err(|_| "it is not JSON".to_owned())?;
        if self == Self::Published {
            require_published_layout(&settings)?;
        }
        let (pointer_dimension_key, base_model_key) = match self {
            Self::Converted => ("pointer_dimension", "base_model"),
            Self::Published => ("head_dim", "base"),
        };
        Ok(KevFolderSettings {
            temperature: positive_temperature(&settings)?,
            pointer_dimension: positive_whole_number(&settings, pointer_dimension_key)?,
            base_model: text_field(&settings, base_model_key).map(str::to_owned),
        })
    }
}

/// A published folder is served only as merged weights with the head where
/// this layout expects it, the only form the engine can load.
fn require_published_layout(settings: &serde_json::Value) -> Result<(), String> {
    let head_file = text_field(settings, "head_file");
    if head_file != Some(KevFolderLayout::Published.head_file_name()) {
        return Err(format!(
            "its head_file is {head_file:?}, not {:?}",
            KevFolderLayout::Published.head_file_name()
        ));
    }
    if text_field(settings, "backbone") != Some("merged") {
        return Err("its backbone is not the merged one".to_owned());
    }
    Ok(())
}

fn is_published_format(format: Option<&str>) -> bool {
    match format {
        Some(format) => format.starts_with(PUBLISHED_FORMAT_PREFIX),
        None => false,
    }
}

fn text_field<'settings>(
    settings: &'settings serde_json::Value,
    key: &str,
) -> Option<&'settings str> {
    settings.get(key).and_then(serde_json::Value::as_str)
}

fn positive_temperature(settings: &serde_json::Value) -> Result<f64, String> {
    match settings
        .get("temperature")
        .and_then(serde_json::Value::as_f64)
    {
        Some(temperature) if temperature.is_finite() && temperature > 0.0 => Ok(temperature),
        _ => Err("its temperature is not a positive number".to_owned()),
    }
}

fn positive_whole_number(settings: &serde_json::Value, key: &str) -> Result<usize, String> {
    match settings.get(key).and_then(serde_json::Value::as_u64) {
        Some(number) if number > 0 => usize::try_from(number)
            .map_err(|_| format!("its {key} does not fit this machine's word size")),
        _ => Err(format!("its {key} is not a positive whole number")),
    }
}

#[cfg(test)]
#[path = "kev_folder_layout_tests.rs"]
mod tests;
