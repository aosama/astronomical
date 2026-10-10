//! Pure planning for image-aware prompt-cache restore and visual row suffixes.

use thiserror::Error;

/// One source image whose persisted or computed visual rows are still needed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen3_5VisualEmbeddingRequiredImage {
    image_index: usize,
    image_visual_embedding_row_count: usize,
    suffix_start_row: usize,
    suffix_row_count: usize,
}

impl Qwen3_5VisualEmbeddingRequiredImage {
    /// Builds a required-image row range in original request image order.
    #[must_use]
    pub const fn new(
        image_index: usize,
        image_visual_embedding_row_count: usize,
        suffix_start_row: usize,
        suffix_row_count: usize,
    ) -> Self {
        Self {
            image_index,
            image_visual_embedding_row_count,
            suffix_start_row,
            suffix_row_count,
        }
    }

    /// Returns the image ordinal from the original request.
    #[must_use]
    pub const fn image_index(&self) -> usize {
        self.image_index
    }

    /// Returns the complete row count for this image before suffix slicing.
    #[must_use]
    pub const fn image_visual_embedding_row_count(&self) -> usize {
        self.image_visual_embedding_row_count
    }

    /// Returns the first row still needed from this image.
    #[must_use]
    pub const fn suffix_start_row(&self) -> usize {
        self.suffix_start_row
    }

    /// Returns how many rows remain needed from this image.
    #[must_use]
    pub const fn suffix_row_count(&self) -> usize {
        self.suffix_row_count
    }
}

/// Visual rows required after an optional prompt-cache restore.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen3_5VisualEmbeddingSuffixPlan {
    restored_visual_embedding_row_count: usize,
    remaining_visual_embedding_row_count: usize,
    required_images: Vec<Qwen3_5VisualEmbeddingRequiredImage>,
}

impl Qwen3_5VisualEmbeddingSuffixPlan {
    /// Returns how many image-pad rows are already represented by restored decoder state.
    #[must_use]
    pub const fn restored_visual_embedding_row_count(&self) -> usize {
        self.restored_visual_embedding_row_count
    }

    /// Returns how many visual rows remain for suffix prefill.
    #[must_use]
    pub const fn remaining_visual_embedding_row_count(&self) -> usize {
        self.remaining_visual_embedding_row_count
    }

    /// Returns required images in original request order, preserving duplicates.
    #[must_use]
    pub fn required_images(&self) -> &[Qwen3_5VisualEmbeddingRequiredImage] {
        &self.required_images
    }
}

/// Plans visual embedding rows needed after restoring a prompt prefix.
pub fn plan_qwen3_5_visual_embedding_suffix(
    prompt_token_ids: &[u32],
    restored_token_count: usize,
    ordered_image_visual_embedding_row_counts: &[usize],
    image_pad_token_id: u32,
) -> Result<Qwen3_5VisualEmbeddingSuffixPlan, Qwen3_5VisualEmbeddingSuffixPlanError> {
    if restored_token_count > prompt_token_ids.len() {
        return Err(
            Qwen3_5VisualEmbeddingSuffixPlanError::RestoredTokenCountExceedsPrompt {
                restored_token_count,
                prompt_token_count: prompt_token_ids.len(),
            },
        );
    }
    if let Some((image_index, _)) = ordered_image_visual_embedding_row_counts
        .iter()
        .enumerate()
        .find(|(_, image_visual_embedding_row_count)| **image_visual_embedding_row_count == 0)
    {
        return Err(Qwen3_5VisualEmbeddingSuffixPlanError::ZeroImageRows { image_index });
    }
    let prompt_image_pad_token_count = prompt_token_ids
        .iter()
        .filter(|token_id| **token_id == image_pad_token_id)
        .count();
    let total_visual_embedding_row_count = ordered_image_visual_embedding_row_counts
        .iter()
        .try_fold(
            0_usize,
            |total_visual_embedding_row_count, image_row_count| {
                total_visual_embedding_row_count.checked_add(*image_row_count)
            },
        )
        .ok_or(Qwen3_5VisualEmbeddingSuffixPlanError::ImageRowCountOverflow)?;
    if prompt_image_pad_token_count != total_visual_embedding_row_count {
        return Err(
            Qwen3_5VisualEmbeddingSuffixPlanError::ImagePadCountMismatch {
                prompt_image_pad_token_count,
                total_visual_embedding_row_count,
            },
        );
    }

    let restored_visual_embedding_row_count = prompt_token_ids[..restored_token_count]
        .iter()
        .filter(|token_id| **token_id == image_pad_token_id)
        .count();
    let remaining_visual_embedding_row_count =
        total_visual_embedding_row_count.saturating_sub(restored_visual_embedding_row_count);
    let mut remaining_restored_visual_embedding_row_count = restored_visual_embedding_row_count;
    let mut required_images = Vec::with_capacity(ordered_image_visual_embedding_row_counts.len());
    for (image_index, image_visual_embedding_row_count) in ordered_image_visual_embedding_row_counts
        .iter()
        .copied()
        .enumerate()
    {
        if remaining_restored_visual_embedding_row_count >= image_visual_embedding_row_count {
            remaining_restored_visual_embedding_row_count -= image_visual_embedding_row_count;
            continue;
        }
        let suffix_start_row = remaining_restored_visual_embedding_row_count;
        let suffix_row_count = image_visual_embedding_row_count - suffix_start_row;
        required_images.push(Qwen3_5VisualEmbeddingRequiredImage::new(
            image_index,
            image_visual_embedding_row_count,
            suffix_start_row,
            suffix_row_count,
        ));
        remaining_restored_visual_embedding_row_count = 0;
    }
    Ok(Qwen3_5VisualEmbeddingSuffixPlan {
        restored_visual_embedding_row_count,
        remaining_visual_embedding_row_count,
        required_images,
    })
}

/// One invalid visual suffix-planning input.
#[derive(Debug, Error, Eq, PartialEq)]
pub enum Qwen3_5VisualEmbeddingSuffixPlanError {
    #[error(
        "restored token count {restored_token_count} exceeds prompt length {prompt_token_count}"
    )]
    RestoredTokenCountExceedsPrompt {
        restored_token_count: usize,
        prompt_token_count: usize,
    },
    #[error("image {image_index} has zero visual rows")]
    ZeroImageRows { image_index: usize },
    #[error("total image visual row count overflowed")]
    ImageRowCountOverflow,
    #[error(
        "prompt has {prompt_image_pad_token_count} image-pad tokens but images provide {total_visual_embedding_row_count} visual rows"
    )]
    ImagePadCountMismatch {
        prompt_image_pad_token_count: usize,
        total_visual_embedding_row_count: usize,
    },
}
