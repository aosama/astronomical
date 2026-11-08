//! How a FLUX.2 Klein artifact stores its weights, and the validated weight
//! files each component owns until it maps them. Kept free of MLX so artifact
//! validation can describe the layout without the runtime.

use std::fs::File;
use std::path::Path;

use crate::artifact_validation::validated_artifact::ValidatedWeightsFile;

/// The file only the 4-bit build has at its root level: the BF16 build is a
/// Diffusers pipeline and always carries `model_index.json`.
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
    #[must_use]
    pub fn detect(model_directory: &Path) -> Self {
        if model_directory
            .join(DIFFUSERS_PIPELINE_INDEX_FILE_NAME)
            .is_file()
        {
            Self::DiffusersBf16
        } else {
            Self::MlxAffine4Bit
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

    #[cfg(feature = "direct-mlx")]
    pub(crate) fn try_clone(&self) -> std::io::Result<Self> {
        Ok(Self {
            layout: self.layout,
            component: self.component,
            shard_files: self.clone_shard_files()?,
            size_bytes: self.size_bytes,
        })
    }
}
