use std::collections::{BTreeMap, BTreeSet};

use thiserror::Error;

/// Opaque identity for one validated SafeTensors source.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct TensorSourceId(u32);

impl TensorSourceId {
    /// Creates an opaque source identity assigned by artifact discovery.
    #[must_use]
    pub const fn new(source_number: u32) -> Self {
        Self(source_number)
    }
}

/// Architecture-neutral semantic ownership of a tensor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TensorSemanticRole {
    Target,
    MultiTokenPrediction,
    Vision,
}

/// Boundary that declared one tensor location.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TensorDeclarationOrigin {
    MainIndex,
    ArchitectureSidecar,
}

/// Optional feature that atomically owns a set of tensor locations.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TensorFeature {
    MultiTokenPrediction,
}

/// Canonical and physical identity for one validated tensor location.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorLocation {
    canonical_name: String,
    stored_name: String,
    source_id: TensorSourceId,
    semantic_role: TensorSemanticRole,
    declaration_origin: TensorDeclarationOrigin,
    feature: Option<TensorFeature>,
}

impl TensorLocation {
    /// Creates a tensor location after architecture-specific name parsing.
    #[must_use]
    pub fn new(
        canonical_name: impl Into<String>,
        stored_name: impl Into<String>,
        source_id: TensorSourceId,
        semantic_role: TensorSemanticRole,
        declaration_origin: TensorDeclarationOrigin,
        feature: Option<TensorFeature>,
    ) -> Self {
        Self {
            canonical_name: canonical_name.into(),
            stored_name: stored_name.into(),
            source_id,
            semantic_role,
            declaration_origin,
            feature,
        }
    }

    #[must_use]
    pub fn canonical_name(&self) -> &str {
        &self.canonical_name
    }

    #[must_use]
    pub fn stored_name(&self) -> &str {
        &self.stored_name
    }

    #[must_use]
    pub const fn source_id(&self) -> TensorSourceId {
        self.source_id
    }

    #[must_use]
    pub const fn semantic_role(&self) -> TensorSemanticRole {
        self.semantic_role
    }

    #[must_use]
    pub const fn declaration_origin(&self) -> TensorDeclarationOrigin {
        self.declaration_origin
    }

    #[must_use]
    pub const fn feature(&self) -> Option<TensorFeature> {
        self.feature
    }
}

/// Convention-neutral canonical inventory for validated tensor locations.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TensorInventory {
    locations_by_canonical_name: BTreeMap<String, TensorLocation>,
    canonical_name_by_physical_location: BTreeMap<(TensorSourceId, String), String>,
}

impl TensorInventory {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            locations_by_canonical_name: BTreeMap::new(),
            canonical_name_by_physical_location: BTreeMap::new(),
        }
    }

    /// Adds one location while rejecting canonical and physical ambiguity.
    pub fn insert(&mut self, location: TensorLocation) -> Result<(), TensorInventoryError> {
        if self
            .locations_by_canonical_name
            .contains_key(location.canonical_name())
        {
            return Err(TensorInventoryError::CanonicalNameCollision {
                canonical_name: location.canonical_name().to_owned(),
            });
        }
        let physical_location = (location.source_id(), location.stored_name().to_owned());
        if self
            .canonical_name_by_physical_location
            .contains_key(&physical_location)
        {
            return Err(TensorInventoryError::PhysicalLocationCollision {
                source_id: location.source_id(),
                stored_name: location.stored_name().to_owned(),
            });
        }
        self.canonical_name_by_physical_location
            .insert(physical_location, location.canonical_name().to_owned());
        self.locations_by_canonical_name
            .insert(location.canonical_name().to_owned(), location);
        Ok(())
    }

    #[must_use]
    pub fn location(&self, canonical_name: &str) -> Option<&TensorLocation> {
        self.locations_by_canonical_name.get(canonical_name)
    }

    pub fn locations(&self) -> impl Iterator<Item = &TensorLocation> {
        self.locations_by_canonical_name.values()
    }

    pub fn source_ids(&self) -> impl Iterator<Item = TensorSourceId> + '_ {
        self.locations_by_canonical_name
            .values()
            .map(TensorLocation::source_id)
            .collect::<BTreeSet<_>>()
            .into_iter()
    }

    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.locations_by_canonical_name.len()
    }

    /// Removes every location owned by an unavailable optional feature.
    pub fn remove_feature(&mut self, feature: TensorFeature) {
        self.locations_by_canonical_name
            .retain(|_, location| location.feature() != Some(feature));
        self.canonical_name_by_physical_location
            .retain(|_, canonical_name| {
                self.locations_by_canonical_name
                    .contains_key(canonical_name)
            });
    }
}

/// Inventory ambiguity detected before runtime tensor allocation.
#[derive(Debug, Error, Eq, PartialEq)]
pub enum TensorInventoryError {
    #[error("canonical tensor name collision for {canonical_name}")]
    CanonicalNameCollision { canonical_name: String },
    #[error("physical tensor location collision for {stored_name}")]
    PhysicalLocationCollision {
        source_id: TensorSourceId,
        stored_name: String,
    },
}
