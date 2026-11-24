//! What discovery knows about a model family: its identifier, its kind, how
//! a folder of it is recognized and how it is discovered (issue #2473,
//! Step 5).
//!
//! The registry holds one `ModelFamilyDescription` per family, in
//! classification order, so discovery asks the descriptions instead of
//! matching on a closed list of families.

use std::path::Path;

use super::DiscoveredModel;

/// The kind of work a model does, which decides the engine and the requests
/// it serves.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ModelKind {
    Chat,
    ImageGeneration,
    Embedding,
    Decision,
    Speech,
}

/// One model family, named by its identifier: the spelling discovery,
/// the catalog and the worker wire share, such as `kokoro`.
///
/// It holds only the identifier, so a family's constant compares and
/// matches like a value; its description is looked up in the registry.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ModelFamily {
    identifier: &'static str,
}

impl ModelFamily {
    /// The family with this identifier. A family declares its own constant
    /// with it, and the registry holds its description.
    #[must_use]
    pub const fn named(identifier: &'static str) -> Self {
        Self { identifier }
    }

    /// The family's identifier.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        self.identifier
    }
}

/// Recognizes a family from a Diffusers pipeline index document.
pub type PipelineIndexRecognizer = fn(&[u8]) -> Result<bool, serde_json::Error>;

/// Recognizes a family from a marker file in its folder.
pub type DirectoryRecognizer = fn(&Path) -> bool;

/// Recognizes a family from its `config.json` `model_type`.
pub type ModelTypeRecognizer = fn(Option<&str>) -> bool;

/// Projects a folder of the family into the neutral discovery contract.
pub type ModelDiscoverer = fn(&Path, &str) -> Option<DiscoveredModel>;

/// Finds the further models a discovered repository bundles in subfolders.
pub type BundledModelDiscoverer = fn(
    &super::model_family_registry::ModelFamilyRegistry,
    &DiscoveredModel,
) -> Vec<DiscoveredModel>;

/// Everything discovery needs to know about one family.
pub struct ModelFamilyDescription {
    pub model_family: ModelFamily,
    pub kind: ModelKind,
    /// Asked of a folder with a `model_index.json`.
    pub recognizes_pipeline_index: Option<PipelineIndexRecognizer>,
    /// Asked of a folder without one, before its `config.json` is read.
    pub recognizes_directory: Option<DirectoryRecognizer>,
    /// Asked of the `model_type` in `config.json` when no marker matched.
    pub recognizes_model_type: Option<ModelTypeRecognizer>,
    /// Asked of every scanned folder: a folder it marks is a pipeline root
    /// although it has no `model_index.json`, so discovery never treats its
    /// component folders as models of their own.
    pub marks_pipeline_root: Option<DirectoryRecognizer>,
    /// The file whose download metadata records the revision of a folder
    /// without a pipeline index.
    pub revision_file_name: &'static str,
    pub discover_model: ModelDiscoverer,
    pub discover_bundled_models: Option<BundledModelDiscoverer>,
}

/// The file a folder without a pipeline index records its revision against.
const CONFIGURATION_FILE_NAME: &str = "config.json";

impl ModelFamilyDescription {
    const fn recognized_by(
        model_family: ModelFamily,
        kind: ModelKind,
        discover_model: ModelDiscoverer,
    ) -> Self {
        Self {
            model_family,
            kind,
            recognizes_pipeline_index: None,
            recognizes_directory: None,
            recognizes_model_type: None,
            marks_pipeline_root: None,
            revision_file_name: CONFIGURATION_FILE_NAME,
            discover_model,
            discover_bundled_models: None,
        }
    }

    /// A family its `config.json` `model_type` names.
    #[must_use]
    pub const fn recognized_by_model_type(
        model_family: ModelFamily,
        kind: ModelKind,
        recognizes_model_type: ModelTypeRecognizer,
        discover_model: ModelDiscoverer,
    ) -> Self {
        let mut family_description = Self::recognized_by(model_family, kind, discover_model);
        family_description.recognizes_model_type = Some(recognizes_model_type);
        family_description
    }

    /// A family a marker file in its folder names.
    #[must_use]
    pub const fn recognized_by_marker_file(
        model_family: ModelFamily,
        kind: ModelKind,
        recognizes_directory: DirectoryRecognizer,
        discover_model: ModelDiscoverer,
    ) -> Self {
        let mut family_description = Self::recognized_by(model_family, kind, discover_model);
        family_description.recognizes_directory = Some(recognizes_directory);
        family_description
    }

    /// A family its Diffusers pipeline index names.
    #[must_use]
    pub const fn recognized_by_pipeline_index(
        model_family: ModelFamily,
        kind: ModelKind,
        recognizes_pipeline_index: PipelineIndexRecognizer,
        discover_model: ModelDiscoverer,
    ) -> Self {
        let mut family_description = Self::recognized_by(model_family, kind, discover_model);
        family_description.recognizes_pipeline_index = Some(recognizes_pipeline_index);
        family_description
    }

    /// Also recognizes a build of the family that ships without a pipeline
    /// index, by its marker file, and reads that build's revision from
    /// `revision_file_name`.
    #[must_use]
    pub const fn also_recognized_by_marker_file(
        mut self,
        recognizes_directory: DirectoryRecognizer,
        revision_file_name: &'static str,
    ) -> Self {
        self.recognizes_directory = Some(recognizes_directory);
        self.revision_file_name = revision_file_name;
        self
    }

    /// Marks a folder as a pipeline root by a file a build of the family
    /// keeps there in place of a pipeline index.
    #[must_use]
    pub const fn also_marks_pipeline_root(
        mut self,
        marks_pipeline_root: DirectoryRecognizer,
    ) -> Self {
        self.marks_pipeline_root = Some(marks_pipeline_root);
        self
    }

    /// Whether this family marks the folder as a pipeline root.
    #[must_use]
    pub fn marks_pipeline_root(&self, directory: &Path) -> bool {
        self.marks_pipeline_root
            .is_some_and(|marks_pipeline_root| marks_pipeline_root(directory))
    }

    /// Whether this family claims a Diffusers pipeline index document.
    ///
    /// # Errors
    /// Returns the document's parse failure from a family that reads it.
    pub fn claims_pipeline_index(
        &self,
        pipeline_index_bytes: &[u8],
    ) -> Result<bool, serde_json::Error> {
        match self.recognizes_pipeline_index {
            Some(recognizes_pipeline_index) => recognizes_pipeline_index(pipeline_index_bytes),
            None => Ok(false),
        }
    }

    /// Whether this family claims a folder by a marker file in it.
    #[must_use]
    pub fn claims_directory(&self, model_directory: &Path) -> bool {
        self.recognizes_directory
            .is_some_and(|recognizes_directory| recognizes_directory(model_directory))
    }

    /// Whether this family claims a `config.json` `model_type`.
    #[must_use]
    pub fn claims_model_type(&self, model_type: Option<&str>) -> bool {
        self.recognizes_model_type
            .is_some_and(|recognizes_model_type| recognizes_model_type(model_type))
    }

    /// Also discovers the models a repository of the family bundles.
    #[must_use]
    pub const fn with_bundled_models(
        mut self,
        discover_bundled_models: BundledModelDiscoverer,
    ) -> Self {
        self.discover_bundled_models = Some(discover_bundled_models);
        self
    }
}

#[cfg(test)]
#[path = "model_family_description_tests.rs"]
mod model_family_description_tests;
