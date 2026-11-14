//! The limit a model runs under and the sizes of its parts.
//!
//! Loading measures a model's geometry once, and the limit comes from
//! configuration or a live change. The memory governor (`crate::ledger`)
//! decides everything about how the limit is used (issue #2221); this type
//! only carries the two facts both engines read when they open it and when
//! they report status before it has opened.

use crate::budget::ram_values::{MlxRamBudgetError, MlxRamBudgetModelGeometry};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MlxRamBudget {
    mlx_active_memory_ceiling_bytes: u64,
    model_geometry: MlxRamBudgetModelGeometry,
}

impl MlxRamBudget {
    /// Fails for a zero limit, under which nothing can run.
    pub const fn new(
        mlx_active_memory_ceiling_bytes: u64,
        model_geometry: MlxRamBudgetModelGeometry,
    ) -> Result<Self, MlxRamBudgetError> {
        Self::with_positive_limit(mlx_active_memory_ceiling_bytes, model_geometry)
    }

    const fn with_positive_limit(
        mlx_active_memory_ceiling_bytes: u64,
        model_geometry: MlxRamBudgetModelGeometry,
    ) -> Result<Self, MlxRamBudgetError> {
        if mlx_active_memory_ceiling_bytes == 0 {
            return Err(MlxRamBudgetError::InvalidCeiling);
        }
        Ok(Self {
            mlx_active_memory_ceiling_bytes,
            model_geometry,
        })
    }

    /// Fails for a zero limit and leaves the current one in place.
    pub const fn update_mlx_active_memory_ceiling_bytes(
        &mut self,
        mlx_active_memory_ceiling_bytes: u64,
    ) -> Result<(), MlxRamBudgetError> {
        if mlx_active_memory_ceiling_bytes == 0 {
            return Err(MlxRamBudgetError::InvalidCeiling);
        }
        self.mlx_active_memory_ceiling_bytes = mlx_active_memory_ceiling_bytes;
        Ok(())
    }

    #[must_use]
    pub const fn mlx_active_memory_ceiling_bytes(&self) -> u64 {
        self.mlx_active_memory_ceiling_bytes
    }

    #[must_use]
    pub const fn model_geometry(&self) -> MlxRamBudgetModelGeometry {
        self.model_geometry
    }
}

#[cfg(test)]
#[path = "ram_tests.rs"]
mod tests;
