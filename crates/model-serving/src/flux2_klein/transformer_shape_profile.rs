//! Model-configured transformer tensor shapes used before MLX allocation.

use std::collections::BTreeMap;

use super::Flux2KleinTransformerConfig;

impl Flux2KleinTransformerConfig {
    pub fn expected_weight_shapes(&self) -> impl Iterator<Item = (String, Vec<usize>)> {
        let hidden_width = self.hidden_width();
        let feed_forward_width = self.feed_forward_width();
        let mut shapes = BTreeMap::from([
            (
                "x_embedder.weight".to_owned(),
                vec![hidden_width, self.input_width()],
            ),
            (
                "context_embedder.weight".to_owned(),
                vec![hidden_width, self.conditioning_width()],
            ),
            (
                "time_guidance_embed.timestep_embedder.linear_1.weight".to_owned(),
                vec![hidden_width, 256],
            ),
            (
                "time_guidance_embed.timestep_embedder.linear_2.weight".to_owned(),
                vec![hidden_width, hidden_width],
            ),
            (
                "double_stream_modulation_img.linear.weight".to_owned(),
                vec![hidden_width * 6, hidden_width],
            ),
            (
                "double_stream_modulation_txt.linear.weight".to_owned(),
                vec![hidden_width * 6, hidden_width],
            ),
            (
                "single_stream_modulation.linear.weight".to_owned(),
                vec![hidden_width * 3, hidden_width],
            ),
            (
                "norm_out.linear.weight".to_owned(),
                vec![hidden_width * 2, hidden_width],
            ),
            (
                "proj_out.weight".to_owned(),
                vec![self.output_width(), hidden_width],
            ),
        ]);
        let matrix_suffixes = [
            "attn.add_k_proj.weight",
            "attn.add_q_proj.weight",
            "attn.add_v_proj.weight",
            "attn.to_add_out.weight",
            "attn.to_k.weight",
            "attn.to_out.0.weight",
            "attn.to_q.weight",
            "attn.to_v.weight",
        ];
        for block_index in 0..self.double_stream_block_count() {
            let prefix = format!("transformer_blocks.{block_index}");
            for suffix in matrix_suffixes {
                shapes.insert(
                    format!("{prefix}.{suffix}"),
                    vec![hidden_width, hidden_width],
                );
            }
            for suffix in [
                "attn.norm_added_k.weight",
                "attn.norm_added_q.weight",
                "attn.norm_k.weight",
                "attn.norm_q.weight",
            ] {
                shapes.insert(
                    format!("{prefix}.{suffix}"),
                    vec![self.attention_head_width()],
                );
            }
            for stream in ["ff", "ff_context"] {
                shapes.insert(
                    format!("{prefix}.{stream}.linear_in.weight"),
                    vec![feed_forward_width * 2, hidden_width],
                );
                shapes.insert(
                    format!("{prefix}.{stream}.linear_out.weight"),
                    vec![hidden_width, feed_forward_width],
                );
            }
        }
        for block_index in 0..self.single_stream_block_count() {
            let prefix = format!("single_transformer_blocks.{block_index}.attn");
            for suffix in ["norm_k.weight", "norm_q.weight"] {
                shapes.insert(
                    format!("{prefix}.{suffix}"),
                    vec![self.attention_head_width()],
                );
            }
            shapes.insert(
                format!("{prefix}.to_qkv_mlp_proj.weight"),
                vec![hidden_width * 3 + feed_forward_width * 2, hidden_width],
            );
            shapes.insert(
                format!("{prefix}.to_out.weight"),
                vec![hidden_width, hidden_width + feed_forward_width],
            );
        }
        shapes.into_iter()
    }
}
