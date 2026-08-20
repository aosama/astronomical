//! Derives a path-free identity for the complete resolved serving snapshot.

use std::collections::HashMap;

use astronomical_config::{DiscoveredModel, ModelCapabilities, ModelFamily};
use sha2::{Digest, Sha256};

use crate::RuntimeModelPolicy;

/// Path-safe semantic identity for one fully resolved serving snapshot.
pub struct ResolvedConfigurationGeneration;

impl ResolvedConfigurationGeneration {
    pub fn derive(
        document_generation: &str,
        discovered_models: &[DiscoveredModel],
        model_policy_catalog: &HashMap<String, RuntimeModelPolicy>,
        unmatched_model_config_ids: &[String],
    ) -> Result<String, serde_json::Error> {
        let mut generation_digest = Sha256::new();
        update_text(&mut generation_digest, document_generation);

        let mut ordered_discovered_models = discovered_models.iter().collect::<Vec<_>>();
        ordered_discovered_models.sort_by(|left, right| left.model_id.cmp(&right.model_id));
        for discovered_model in ordered_discovered_models {
            update_text(&mut generation_digest, &discovered_model.model_id);
            update_text(&mut generation_digest, &discovered_model.revision);
            update_text(
                &mut generation_digest,
                model_family_identity(discovered_model.model_family),
            );
            update_optional_text(
                &mut generation_digest,
                discovered_model.provider_model_id.as_deref(),
            );
            update_optional_text(
                &mut generation_digest,
                discovered_model
                    .license
                    .map(astronomical_config::ModelLicense::spdx_identifier),
            );
            update_model_capabilities(&mut generation_digest, &discovered_model.capabilities);
            update_number(&mut generation_digest, discovered_model.model_size_bytes);
        }

        let mut ordered_model_policies = model_policy_catalog.iter().collect::<Vec<_>>();
        ordered_model_policies.sort_by(|(left_id, _), (right_id, _)| left_id.cmp(right_id));
        for (model_id, model_policy) in ordered_model_policies {
            update_text(&mut generation_digest, model_id);
            update_number(
                &mut generation_digest,
                model_policy.generation_defaults.maximum_output_tokens,
            );
            update_optional_number(
                &mut generation_digest,
                model_policy.generation_defaults.temperature_thousandths,
            );
            update_optional_number(
                &mut generation_digest,
                model_policy.generation_defaults.top_p_thousandths,
            );
            let worker_policy_bytes = serde_json::to_vec(&model_policy.worker_model_configuration)?;
            update_bytes(&mut generation_digest, &worker_policy_bytes);
        }

        for unmatched_model_id in unmatched_model_config_ids {
            update_text(&mut generation_digest, unmatched_model_id);
        }
        Ok(lowercase_hex(&generation_digest.finalize()))
    }

    /// Identifies the exact live state produced when only memory applies from a larger candidate.
    pub fn derive_memory_only_transition(
        prior_resolved_generation: &str,
        maximum_mlx_memory_bytes: Option<u64>,
    ) -> String {
        let mut generation_digest = Sha256::new();
        update_text(
            &mut generation_digest,
            "astronomical-memory-only-configuration-transition-v1",
        );
        update_text(&mut generation_digest, prior_resolved_generation);
        update_optional_number(&mut generation_digest, maximum_mlx_memory_bytes);
        lowercase_hex(&generation_digest.finalize())
    }
}

fn model_family_identity(model_family: ModelFamily) -> &'static str {
    match model_family {
        ModelFamily::Qwen3_5 => "qwen3_5",
        ModelFamily::Laguna => "laguna",
        ModelFamily::DeepSeekV4 => "deepseek_v4",
        ModelFamily::Flux2Klein => "flux2_klein",
    }
}

fn update_model_capabilities(
    generation_digest: &mut Sha256,
    model_capabilities: &ModelCapabilities,
) {
    match model_capabilities {
        ModelCapabilities::Chat(capabilities) => {
            update_text(generation_digest, "chat");
            update_number(generation_digest, capabilities.context_window);
            update_number(generation_digest, capabilities.max_input_tokens);
            update_number(generation_digest, capabilities.max_output_tokens);
            update_boolean(generation_digest, capabilities.supports_vision);
            update_boolean(generation_digest, capabilities.supports_reasoning);
            update_boolean(generation_digest, capabilities.supports_tool_calls);
        }
        ModelCapabilities::ImageGeneration(capabilities) => {
            update_text(generation_digest, "image_generation");
            update_boolean(generation_digest, capabilities.supports_text_to_image);
            update_boolean(generation_digest, capabilities.supports_image_editing);
            update_boolean(
                generation_digest,
                capabilities.supports_multiple_reference_images,
            );
        }
    }
}

fn lowercase_hex(bytes: &[u8]) -> String {
    const HEX_DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        encoded.push(char::from(HEX_DIGITS[usize::from(byte >> 4)]));
        encoded.push(char::from(HEX_DIGITS[usize::from(byte & 0x0f)]));
    }
    encoded
}

fn update_text(generation_digest: &mut Sha256, text: &str) {
    update_bytes(generation_digest, text.as_bytes());
}

fn update_optional_text(generation_digest: &mut Sha256, text: Option<&str>) {
    match text {
        Some(text) => {
            generation_digest.update([1]);
            update_text(generation_digest, text);
        }
        None => generation_digest.update([0]),
    }
}

fn update_boolean(generation_digest: &mut Sha256, state: bool) {
    generation_digest.update([u8::from(state)]);
}

fn update_bytes(generation_digest: &mut Sha256, bytes: &[u8]) {
    generation_digest.update((bytes.len() as u64).to_le_bytes());
    generation_digest.update(bytes);
}

fn update_number(generation_digest: &mut Sha256, number: impl Into<u64>) {
    generation_digest.update(number.into().to_le_bytes());
}

fn update_optional_number(generation_digest: &mut Sha256, number: Option<impl Into<u64> + Copy>) {
    match number {
        Some(number) => {
            generation_digest.update([1]);
            update_number(generation_digest, number);
        }
        None => generation_digest.update([0]),
    }
}
