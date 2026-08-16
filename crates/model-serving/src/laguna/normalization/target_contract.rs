use super::{
    layer_descriptor::LagunaLayerDescriptor, model_descriptor::LagunaModelDescriptor,
    storage_descriptor::LagunaStorageDescriptor,
};

/// Canonical configuration contract consumed before Laguna model construction.
#[derive(Clone, Debug, PartialEq)]
pub struct LagunaTargetContract {
    model: LagunaModelDescriptor,
    layers: Vec<LagunaLayerDescriptor>,
    storage: LagunaStorageDescriptor,
}

impl LagunaTargetContract {
    pub(super) fn new(
        model: LagunaModelDescriptor,
        layers: Vec<LagunaLayerDescriptor>,
        storage: LagunaStorageDescriptor,
    ) -> Self {
        Self {
            model,
            layers,
            storage,
        }
    }

    /// Returns validated model-wide geometry and behavior.
    #[must_use]
    pub const fn model(&self) -> &LagunaModelDescriptor {
        &self.model
    }

    /// Returns exactly one canonical descriptor for every decoder layer in order.
    #[must_use]
    pub fn layers(&self) -> &[LagunaLayerDescriptor] {
        &self.layers
    }

    /// Returns canonical physical weight storage declared by configuration.
    #[must_use]
    pub const fn storage(&self) -> &LagunaStorageDescriptor {
        &self.storage
    }
}
