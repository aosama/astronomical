//! How a FLUX.2 Klein artifact stores its weights, and the validated weight
//! files each component owns until it maps them. Kept free of MLX so artifact
//! validation can describe the layout without the runtime.

use std::fs::File;
use std::path::Path;

use crate::artifact_validation::validated_artifact::ValidatedWeightsFile;

use super::mlx_4bit_build::MLX_4BIT_TRANSFORMER_INDEX_FILE_NAME;

/// The pipeline index only the BF16 Diffusers build has; the 4-bit build ships none.
const DIFFUSERS_PIPELINE_INDEX_FILE_NAME: &str = "model_index.json";

/// How an artifact stores its weights on disk.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Flux2KleinWeightLayout {
    /// The official BF16 build: Diffusers names, dense BF16 tensors,
    /// PyTorch-layout convolution kernels.
    DiffusersBf16,
    /// The 4-bit MLX build: mflux names, 4-bit affine projections, MLX-layout
    /// convolution kernels.
    MlxAffine4Bit,
}

impl Flux2KleinWeightLayout {
    /// The layout the files in `model_directory` claim. This only chooses which
    /// validator runs; that validator then proves the claim tensor by tensor.
    /// The 4-bit layout needs positive evidence (its transformer index without
    /// a pipeline index); anything else is validated as the BF16 build, so an
    /// incomplete BF16 tree is refused for the file it lacks.
    #[must_use]
    pub fn detect(model_directory: &Path) -> Self {
        let has_pipeline_index = model_directory
            .join(DIFFUSERS_PIPELINE_INDEX_FILE_NAME)
            .is_file();
        let has_4bit_transformer_index = model_directory
            .join(MLX_4BIT_TRANSFORMER_INDEX_FILE_NAME)
            .is_file();
        if has_4bit_transformer_index && !has_pipeline_index {
            Self::MlxAffine4Bit
        } else {
            Self::DiffusersBf16
        }
    }
}

/// The three weight-owning parts of the pipeline.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Flux2KleinComponent {
    Transformer,
    TextEncoder,
    Vae,
}

impl Flux2KleinComponent {
    /// The component's folder in both builds.
    #[must_use]
    pub const fn folder_name(self) -> &'static str {
        match self {
            Self::Transformer => "transformer",
            Self::TextEncoder => "text_encoder",
            Self::Vae => "vae",
        }
    }

    /// Whether the engine multiplies this component's projections on packed
    /// 4-bit weights. The transformer and text encoder bind their projections
    /// with `Flux2KleinTensorStore::weight`; the VAE binds its four attention
    /// projections with `dense`, dequantizing at load so its arithmetic is
    /// unchanged. The 4-bit memory accounting follows this rule, so a change in
    /// how a component binds its projections belongs here too.
    #[must_use]
    pub const fn keeps_projections_packed(self) -> bool {
        !matches!(self, Self::Vae)
    }

    /// The component as rejection messages name it.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Transformer => "transformer",
            Self::TextEncoder => "text encoder",
            Self::Vae => "VAE",
        }
    }
}

/// One component's validated weight shard files and their layout, kept as
/// descriptors until the component maps them.
#[derive(Debug)]
pub struct Flux2KleinComponentWeightFiles {
    layout: Flux2KleinWeightLayout,
    component: Flux2KleinComponent,
    shard_files: Vec<File>,
    size_bytes: u64,
}

impl Flux2KleinComponentWeightFiles {
    #[must_use]
    pub(crate) fn new(
        layout: Flux2KleinWeightLayout,
        component: Flux2KleinComponent,
        shard_files: Vec<ValidatedWeightsFile>,
    ) -> Self {
        let size_bytes = shard_files
            .iter()
            .map(ValidatedWeightsFile::size_bytes)
            .sum();
        Self {
            layout,
            component,
            shard_files: shard_files
                .into_iter()
                .map(ValidatedWeightsFile::into_file)
                .collect(),
            size_bytes,
        }
    }

    #[must_use]
    pub fn shard_count(&self) -> usize {
        self.shard_files.len()
    }

    /// Bytes on disk across every shard, headers included.
    #[must_use]
    pub const fn size_bytes(&self) -> u64 {
        self.size_bytes
    }

    #[must_use]
    pub const fn layout(&self) -> Flux2KleinWeightLayout {
        self.layout
    }

    #[must_use]
    pub const fn component(&self) -> Flux2KleinComponent {
        self.component
    }

    #[cfg(feature = "direct-mlx")]
    /// Fresh descriptors for every shard, so one mapping never consumes the
    /// files another mapping needs.
    pub(crate) fn clone_shard_files(&self) -> std::io::Result<Vec<File>> {
        self.shard_files.iter().map(File::try_clone).collect()
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::Flux2KleinWeightLayout;

    fn write(model_directory: &std::path::Path, relative_path: &str) {
        let file_path = model_directory.join(relative_path);
        fs::create_dir_all(file_path.parent().expect("the file should have a folder"))
            .expect("the folder should be created");
        fs::write(file_path, "{}").expect("the marker file should be written");
    }

    #[test]
    fn should_detect_the_4bit_layout_from_its_transformer_index() {
        let model_directory = tempfile::tempdir().expect("a model folder should exist");
        write(
            model_directory.path(),
            "transformer/model.safetensors.index.json",
        );

        assert_eq!(
            Flux2KleinWeightLayout::detect(model_directory.path()),
            Flux2KleinWeightLayout::MlxAffine4Bit
        );
    }

    #[test]
    fn should_detect_the_bf16_layout_from_its_pipeline_index() {
        let model_directory = tempfile::tempdir().expect("a model folder should exist");
        write(model_directory.path(), "model_index.json");
        write(
            model_directory.path(),
            "transformer/model.safetensors.index.json",
        );

        assert_eq!(
            Flux2KleinWeightLayout::detect(model_directory.path()),
            Flux2KleinWeightLayout::DiffusersBf16
        );
    }

    #[test]
    fn should_validate_a_folder_without_either_index_as_the_bf16_build() {
        let model_directory = tempfile::tempdir().expect("a model folder should exist");

        assert_eq!(
            Flux2KleinWeightLayout::detect(model_directory.path()),
            Flux2KleinWeightLayout::DiffusersBf16
        );
    }
}
