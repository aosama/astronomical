//! Validated model geometry shared by weight binding and graph construction.

use super::Flux2KleinTransformerGeometryError;
use crate::flux2_klein::Flux2KleinTransformerConfig;

#[derive(Clone, Debug, PartialEq)]
pub struct Flux2KleinTransformerGeometry {
    hidden_width: usize,
    attention_head_count: usize,
    attention_head_width: usize,
    input_width: usize,
    conditioning_width: usize,
    timestep_embedding_width: usize,
    feed_forward_width: usize,
    rope_axis_widths: [usize; 4],
    rope_theta: f32,
    double_stream_block_count: usize,
    single_stream_block_count: usize,
    output_width: usize,
    normalization_epsilon: f32,
}

impl Flux2KleinTransformerGeometry {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        hidden_width: usize,
        attention_head_count: usize,
        attention_head_width: usize,
        input_width: usize,
        conditioning_width: usize,
        rope_axis_widths: [usize; 4],
        rope_theta: f32,
        double_stream_block_count: usize,
        single_stream_block_count: usize,
        output_width: usize,
        normalization_epsilon: f32,
    ) -> Result<Self, Flux2KleinTransformerGeometryError> {
        let dimensions = [
            hidden_width,
            attention_head_count,
            attention_head_width,
            input_width,
            conditioning_width,
            double_stream_block_count,
            single_stream_block_count,
            output_width,
        ];
        if dimensions.contains(&0) {
            return Err(Flux2KleinTransformerGeometryError::ZeroDimension);
        }
        if dimensions
            .iter()
            .any(|dimension| *dimension > i32::MAX as usize)
        {
            return Err(Flux2KleinTransformerGeometryError::ArithmeticOverflow);
        }
        if attention_head_count.checked_mul(attention_head_width) != Some(hidden_width) {
            return Err(Flux2KleinTransformerGeometryError::AttentionWidthMismatch);
        }
        if rope_axis_widths
            .iter()
            .any(|width| *width == 0 || width % 2 != 0)
        {
            return Err(Flux2KleinTransformerGeometryError::InvalidRopeAxis);
        }
        let rope_width = rope_axis_widths
            .iter()
            .try_fold(0_usize, |total, width| total.checked_add(*width))
            .ok_or(Flux2KleinTransformerGeometryError::ArithmeticOverflow)?;
        if rope_width != attention_head_width {
            return Err(Flux2KleinTransformerGeometryError::RopeWidthMismatch {
                rope_width,
                head_width: attention_head_width,
            });
        }
        if !rope_theta.is_finite()
            || rope_theta <= 0.0
            || !normalization_epsilon.is_finite()
            || normalization_epsilon <= 0.0
        {
            return Err(Flux2KleinTransformerGeometryError::InvalidFloatingPointConstant);
        }
        let feed_forward_width = hidden_width
            .checked_mul(3)
            .ok_or(Flux2KleinTransformerGeometryError::ArithmeticOverflow)?;
        hidden_width
            .checked_mul(6)
            .and_then(|_| feed_forward_width.checked_mul(2))
            .and_then(|double_feed_forward| {
                hidden_width
                    .checked_mul(3)?
                    .checked_add(double_feed_forward)
            })
            .filter(|largest_width| *largest_width <= i32::MAX as usize)
            .ok_or(Flux2KleinTransformerGeometryError::ArithmeticOverflow)?;
        Ok(Self {
            hidden_width,
            attention_head_count,
            attention_head_width,
            input_width,
            conditioning_width,
            timestep_embedding_width: 256,
            feed_forward_width,
            rope_axis_widths,
            rope_theta,
            double_stream_block_count,
            single_stream_block_count,
            output_width,
            normalization_epsilon,
        })
    }

    pub fn from_config(
        config: &Flux2KleinTransformerConfig,
    ) -> Result<Self, Flux2KleinTransformerGeometryError> {
        Self::new(
            config.hidden_width(),
            config.attention_head_count(),
            config.attention_head_width(),
            config.input_width(),
            config.conditioning_width(),
            config.rope_axis_widths(),
            config.rope_theta() as f32,
            config.double_stream_block_count(),
            config.single_stream_block_count(),
            config.output_width(),
            config.normalization_epsilon() as f32,
        )
    }

    pub const fn hidden_width(&self) -> usize {
        self.hidden_width
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
    pub const fn timestep_embedding_width(&self) -> usize {
        self.timestep_embedding_width
    }
    pub const fn feed_forward_width(&self) -> usize {
        self.feed_forward_width
    }
    pub const fn rope_axis_widths(&self) -> [usize; 4] {
        self.rope_axis_widths
    }
    pub const fn rope_theta(&self) -> f32 {
        self.rope_theta
    }
    pub const fn double_stream_block_count(&self) -> usize {
        self.double_stream_block_count
    }
    pub const fn single_stream_block_count(&self) -> usize {
        self.single_stream_block_count
    }
    pub const fn total_block_count(&self) -> usize {
        self.double_stream_block_count + self.single_stream_block_count
    }
    pub const fn output_width(&self) -> usize {
        self.output_width
    }
    pub const fn normalization_epsilon(&self) -> f32 {
        self.normalization_epsilon
    }

    pub fn block_index_for_weight_name(&self, tensor_name: &str) -> Option<usize> {
        if let Some(suffix) = tensor_name.strip_prefix("transformer_blocks.") {
            return suffix
                .split('.')
                .next()?
                .parse::<usize>()
                .ok()
                .filter(|block_index| *block_index < self.double_stream_block_count);
        }
        tensor_name
            .strip_prefix("single_transformer_blocks.")?
            .split('.')
            .next()?
            .parse::<usize>()
            .ok()
            .filter(|block_index| *block_index < self.single_stream_block_count)
            .map(|block_index| block_index + self.double_stream_block_count)
    }

    pub fn expected_weight_shapes(&self) -> impl Iterator<Item = (String, Vec<usize>)> {
        let hidden = self.hidden_width;
        let feed_forward = self.feed_forward_width;
        let mut shapes = vec![
            (
                "x_embedder.weight".to_owned(),
                vec![hidden, self.input_width],
            ),
            (
                "context_embedder.weight".to_owned(),
                vec![hidden, self.conditioning_width],
            ),
            (
                "time_guidance_embed.timestep_embedder.linear_1.weight".to_owned(),
                vec![hidden, self.timestep_embedding_width],
            ),
            (
                "time_guidance_embed.timestep_embedder.linear_2.weight".to_owned(),
                vec![hidden, hidden],
            ),
            (
                "double_stream_modulation_img.linear.weight".to_owned(),
                vec![hidden * 6, hidden],
            ),
            (
                "double_stream_modulation_txt.linear.weight".to_owned(),
                vec![hidden * 6, hidden],
            ),
            (
                "single_stream_modulation.linear.weight".to_owned(),
                vec![hidden * 3, hidden],
            ),
            (
                "norm_out.linear.weight".to_owned(),
                vec![hidden * 2, hidden],
            ),
            (
                "proj_out.weight".to_owned(),
                vec![self.output_width, hidden],
            ),
        ];
        let double_matrix_suffixes = [
            "attn.add_k_proj.weight",
            "attn.add_q_proj.weight",
            "attn.add_v_proj.weight",
            "attn.to_add_out.weight",
            "attn.to_k.weight",
            "attn.to_out.0.weight",
            "attn.to_q.weight",
            "attn.to_v.weight",
        ];
        for block_index in 0..self.double_stream_block_count {
            let prefix = format!("transformer_blocks.{block_index}");
            shapes.extend(
                double_matrix_suffixes
                    .map(|suffix| (format!("{prefix}.{suffix}"), vec![hidden, hidden])),
            );
            for suffix in [
                "attn.norm_added_k.weight",
                "attn.norm_added_q.weight",
                "attn.norm_k.weight",
                "attn.norm_q.weight",
            ] {
                shapes.push((
                    format!("{prefix}.{suffix}"),
                    vec![self.attention_head_width],
                ));
            }
            for stream in ["ff", "ff_context"] {
                shapes.push((
                    format!("{prefix}.{stream}.linear_in.weight"),
                    vec![feed_forward * 2, hidden],
                ));
                shapes.push((
                    format!("{prefix}.{stream}.linear_out.weight"),
                    vec![hidden, feed_forward],
                ));
            }
        }
        for block_index in 0..self.single_stream_block_count {
            let prefix = format!("single_transformer_blocks.{block_index}.attn");
            shapes.push((
                format!("{prefix}.norm_k.weight"),
                vec![self.attention_head_width],
            ));
            shapes.push((
                format!("{prefix}.norm_q.weight"),
                vec![self.attention_head_width],
            ));
            shapes.push((
                format!("{prefix}.to_qkv_mlp_proj.weight"),
                vec![hidden * 3 + feed_forward * 2, hidden],
            ));
            shapes.push((
                format!("{prefix}.to_out.weight"),
                vec![hidden, hidden + feed_forward],
            ));
        }
        shapes.into_iter()
    }
}
