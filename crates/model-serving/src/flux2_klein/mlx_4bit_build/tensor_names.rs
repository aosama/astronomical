//! The one place that knows how the 4-bit MLX build names its tensors.
//!
//! The rest of the engine speaks the Diffusers names of the BF16 build. The
//! 4-bit build (quantized with mflux) renames a few of them; a tensor-for-
//! tensor comparison of the two builds found exactly these differences, and no
//! others (issue #1904).

use crate::flux2_klein::weight_layout::Flux2KleinComponent;

/// The 4-bit build's name for the Diffusers tensor `diffusers_tensor_name`.
#[must_use]
pub(crate) fn mlx_4bit_tensor_name(
    component: Flux2KleinComponent,
    diffusers_tensor_name: &str,
) -> String {
    match component {
        // The text encoder is a bare Qwen3 model without the `model.` wrapper.
        Flux2KleinComponent::TextEncoder => diffusers_tensor_name
            .strip_prefix("model.")
            .unwrap_or(diffusers_tensor_name)
            .to_owned(),
        // mflux drops the `Sequential` index of attention output projections.
        Flux2KleinComponent::Vae => diffusers_tensor_name.replace(".to_out.0.", ".to_out."),
        Flux2KleinComponent::Transformer => diffusers_tensor_name
            .replace(".to_out.0.", ".to_out.")
            .replace(
                "time_guidance_embed.timestep_embedder.",
                "time_guidance_embed.",
            ),
    }
}

#[cfg(test)]
mod tests {
    use super::mlx_4bit_tensor_name;
    use crate::flux2_klein::weight_layout::Flux2KleinComponent;

    #[test]
    fn should_strip_the_model_prefix_of_text_encoder_tensors() {
        assert_eq!(
            mlx_4bit_tensor_name(
                Flux2KleinComponent::TextEncoder,
                "model.layers.3.mlp.down_proj.weight"
            ),
            "layers.3.mlp.down_proj.weight"
        );
        assert_eq!(
            mlx_4bit_tensor_name(
                Flux2KleinComponent::TextEncoder,
                "model.embed_tokens.weight"
            ),
            "embed_tokens.weight"
        );
    }

    #[test]
    fn should_drop_the_sequential_index_of_attention_outputs() {
        assert_eq!(
            mlx_4bit_tensor_name(
                Flux2KleinComponent::Transformer,
                "transformer_blocks.4.attn.to_out.0.weight"
            ),
            "transformer_blocks.4.attn.to_out.weight"
        );
        assert_eq!(
            mlx_4bit_tensor_name(
                Flux2KleinComponent::Vae,
                "decoder.mid_block.attentions.0.to_out.0.bias"
            ),
            "decoder.mid_block.attentions.0.to_out.bias"
        );
    }

    #[test]
    fn should_flatten_the_timestep_embedder_of_the_transformer() {
        assert_eq!(
            mlx_4bit_tensor_name(
                Flux2KleinComponent::Transformer,
                "time_guidance_embed.timestep_embedder.linear_2.weight"
            ),
            "time_guidance_embed.linear_2.weight"
        );
    }

    #[test]
    fn should_keep_every_other_name_unchanged() {
        for (component, unchanged_name) in [
            (
                Flux2KleinComponent::Transformer,
                "single_transformer_blocks.0.attn.to_out.weight",
            ),
            (Flux2KleinComponent::Transformer, "proj_out.weight"),
            (Flux2KleinComponent::Vae, "decoder.conv_in.weight"),
            (Flux2KleinComponent::Vae, "bn.running_mean"),
            (Flux2KleinComponent::TextEncoder, "norm.weight"),
        ] {
            assert_eq!(
                mlx_4bit_tensor_name(component, unchanged_name),
                unchanged_name
            );
        }
    }
}
