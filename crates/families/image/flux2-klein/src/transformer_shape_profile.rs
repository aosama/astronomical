//! Model-configured transformer tensor shapes used before MLX allocation.

use std::collections::BTreeMap;

use super::Flux2KleinTransformerConfig;

/// The eight attention weights of a double-stream block whose shape is hidden width by hidden
/// width. The execution geometry lists the same names, so both tables share this one list.
pub(crate) const DOUBLE_STREAM_MATRIX_SUFFIXES: [&str; 8] = [
    "attn.add_k_proj.weight",
    "attn.add_q_proj.weight",
    "attn.add_v_proj.weight",
    "attn.to_add_out.weight",
    "attn.to_k.weight",
    "attn.to_out.0.weight",
    "attn.to_q.weight",
    "attn.to_v.weight",
];

/// The four query and key normalization weights of a double-stream block, one attention head
/// wide.
pub(crate) const DOUBLE_STREAM_HEAD_NORM_SUFFIXES: [&str; 4] = [
    "attn.norm_added_k.weight",
    "attn.norm_added_q.weight",
    "attn.norm_k.weight",
    "attn.norm_q.weight",
];

impl Flux2KleinTransformerConfig {
    pub fn expected_weight_shapes(&self) -> impl Iterator<Item = (String, Vec<usize>)> {
        let mut shapes = self.shared_weight_shapes();
        for block_index in 0..self.double_stream_block_count() {
            self.insert_double_stream_block_shapes(&mut shapes, block_index);
        }
        for block_index in 0..self.single_stream_block_count() {
            self.insert_single_stream_block_shapes(&mut shapes, block_index);
        }
        shapes.into_iter()
    }

    /// The nine weights that sit outside every block.
    fn shared_weight_shapes(&self) -> BTreeMap<String, Vec<usize>> {
        let hidden_width = self.hidden_width();
        BTreeMap::from([
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
        ])
    }

    fn insert_double_stream_block_shapes(
        &self,
        shapes: &mut BTreeMap<String, Vec<usize>>,
        block_index: usize,
    ) {
        let hidden_width = self.hidden_width();
        let feed_forward_width = self.feed_forward_width();
        let prefix = format!("transformer_blocks.{block_index}");
        for suffix in DOUBLE_STREAM_MATRIX_SUFFIXES {
            shapes.insert(
                format!("{prefix}.{suffix}"),
                vec![hidden_width, hidden_width],
            );
        }
        for suffix in DOUBLE_STREAM_HEAD_NORM_SUFFIXES {
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

    fn insert_single_stream_block_shapes(
        &self,
        shapes: &mut BTreeMap<String, Vec<usize>>,
        block_index: usize,
    ) {
        let hidden_width = self.hidden_width();
        let feed_forward_width = self.feed_forward_width();
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
}

#[cfg(test)]
#[path = "transformer_shape_profile_tests.rs"]
mod tests;
