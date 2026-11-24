//! The families discovery recognizes, handed to it rather than known by it
//! (issue #2473, Step 5).
//!
//! Discovery holds the rules every family shares: how a folder is walked,
//! read and classified. Which families exist is a list in the
//! `model-family-registry` crate, above this one, so a family's own
//! recognizers can live next to the family and this crate never names one.
//! Every discovery entry point takes the registry, and the registry crate
//! offers the same entry points with its list already passed in.

use std::path::Path;

use super::model_family_description::{ModelFamily, ModelFamilyDescription, ModelKind};

/// Every registered family's description, in classification order: within
/// each way of recognizing a folder (pipeline index, marker file,
/// `model_type`), the first family that claims it wins.
#[derive(Clone, Copy)]
pub struct ModelFamilyRegistry {
    registered_families: &'static [ModelFamilyDescription],
}

impl ModelFamilyRegistry {
    #[must_use]
    pub const fn new(registered_families: &'static [ModelFamilyDescription]) -> Self {
        Self {
            registered_families,
        }
    }

    /// Every description, in classification order.
    #[must_use]
    pub const fn descriptions(&self) -> &'static [ModelFamilyDescription] {
        self.registered_families
    }

    /// The description of the family with this identifier.
    #[must_use]
    pub fn description_named(&self, identifier: &str) -> Option<&'static ModelFamilyDescription> {
        self.registered_families
            .iter()
            .find(|registered_description| {
                registered_description.model_family.as_str() == identifier
            })
    }

    /// The registered family with this identifier.
    #[must_use]
    pub fn family_named(&self, identifier: &str) -> Option<ModelFamily> {
        self.description_named(identifier)
            .map(|description| description.model_family)
    }

    /// The family's description, or `None` for a family nothing registers.
    #[must_use]
    pub fn description_of(
        &self,
        model_family: ModelFamily,
    ) -> Option<&'static ModelFamilyDescription> {
        self.description_named(model_family.as_str())
    }

    /// The kind of model the family is, or `None` when it is not registered.
    #[must_use]
    pub fn kind_of(&self, model_family: ModelFamily) -> Option<ModelKind> {
        self.description_of(model_family)
            .map(|description| description.kind)
    }

    /// The first family that claims this `config.json` `model_type`, without
    /// claiming the folder holds a servable model.
    #[must_use]
    pub fn family_of_model_type(&self, model_type: Option<&str>) -> Option<ModelFamily> {
        self.registered_families
            .iter()
            .find(|family_description| family_description.claims_model_type(model_type))
            .map(|family_description| family_description.model_family)
    }

    /// The first family whose marker file the folder carries.
    #[must_use]
    pub fn family_marked_by_file(&self, model_directory: &Path) -> Option<ModelFamily> {
        self.registered_families
            .iter()
            .find(|family_description| family_description.claims_directory(model_directory))
            .map(|family_description| family_description.model_family)
    }

    /// Whether a family marks this folder as a pipeline root that carries no
    /// pipeline index of its own, as the 4-bit FLUX.2 Klein build does.
    #[must_use]
    pub fn marks_pipeline_root(&self, directory: &Path) -> bool {
        self.registered_families
            .iter()
            .any(|family_description| family_description.marks_pipeline_root(directory))
    }
}
