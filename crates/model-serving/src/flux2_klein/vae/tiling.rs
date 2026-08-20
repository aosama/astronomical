//! Tiled decode geometry assigns every output core to one tile while retaining halo context.

use super::Flux2KleinVaeError;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Flux2KleinVaeTilingConfig {
    owned_latent_edge: usize,
    overlap_latents: usize,
}

impl Flux2KleinVaeTilingConfig {
    pub fn new(
        owned_latent_edge: usize,
        overlap_latents: usize,
    ) -> Result<Self, Flux2KleinVaeError> {
        if owned_latent_edge == 0 {
            return Err(Flux2KleinVaeError::tiling_geometry(
                "owned latent edge must be positive",
            ));
        }
        Ok(Self {
            owned_latent_edge,
            overlap_latents,
        })
    }

    #[must_use]
    pub const fn owned_latent_edge(&self) -> usize {
        self.owned_latent_edge
    }

    #[must_use]
    pub const fn overlap_latents(&self) -> usize {
        self.overlap_latents
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Flux2KleinVaeTile {
    source_row_start: usize,
    source_row_end: usize,
    source_column_start: usize,
    source_column_end: usize,
    owned_row_start: usize,
    owned_row_end: usize,
    owned_column_start: usize,
    owned_column_end: usize,
}

impl Flux2KleinVaeTile {
    pub const fn source_row_start(&self) -> usize {
        self.source_row_start
    }
    pub const fn source_row_end(&self) -> usize {
        self.source_row_end
    }
    pub const fn source_column_start(&self) -> usize {
        self.source_column_start
    }
    pub const fn source_column_end(&self) -> usize {
        self.source_column_end
    }
    pub const fn owned_row_start(&self) -> usize {
        self.owned_row_start
    }
    pub const fn owned_row_end(&self) -> usize {
        self.owned_row_end
    }
    pub const fn owned_column_start(&self) -> usize {
        self.owned_column_start
    }
    pub const fn owned_column_end(&self) -> usize {
        self.owned_column_end
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Flux2KleinVaeTilePlan {
    tiles: Vec<Flux2KleinVaeTile>,
}

impl Flux2KleinVaeTilePlan {
    pub fn new(
        latent_width: usize,
        latent_height: usize,
        config: Flux2KleinVaeTilingConfig,
    ) -> Result<Self, Flux2KleinVaeError> {
        if latent_width == 0 || latent_height == 0 {
            return Err(Flux2KleinVaeError::tiling_geometry(
                "latent width and height must be positive",
            ));
        }
        let row_count = latent_height.div_ceil(config.owned_latent_edge);
        let column_count = latent_width.div_ceil(config.owned_latent_edge);
        let capacity = row_count
            .checked_mul(column_count)
            .ok_or_else(|| Flux2KleinVaeError::tiling_geometry("tile count overflow"))?;
        let mut tiles = Vec::with_capacity(capacity);
        for owned_row_start in (0..latent_height).step_by(config.owned_latent_edge) {
            let owned_row_end = owned_row_start
                .saturating_add(config.owned_latent_edge)
                .min(latent_height);
            for owned_column_start in (0..latent_width).step_by(config.owned_latent_edge) {
                let owned_column_end = owned_column_start
                    .saturating_add(config.owned_latent_edge)
                    .min(latent_width);
                tiles.push(Flux2KleinVaeTile {
                    source_row_start: owned_row_start.saturating_sub(config.overlap_latents),
                    source_row_end: owned_row_end
                        .saturating_add(config.overlap_latents)
                        .min(latent_height),
                    source_column_start: owned_column_start.saturating_sub(config.overlap_latents),
                    source_column_end: owned_column_end
                        .saturating_add(config.overlap_latents)
                        .min(latent_width),
                    owned_row_start,
                    owned_row_end,
                    owned_column_start,
                    owned_column_end,
                });
            }
        }
        Ok(Self { tiles })
    }

    #[must_use]
    pub fn tiles(&self) -> &[Flux2KleinVaeTile] {
        &self.tiles
    }
}
