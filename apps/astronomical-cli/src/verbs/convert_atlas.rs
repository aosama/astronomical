//! `astronomical models convert-atlas MODEL [--output-model-id ORG/NAME]`:
//! turns a mixture-of-experts model into an Expert Atlas model of its own.
//!
//! MODEL is a Library model id (`organization/model-name`) or a model
//! folder. The new model lands in this instance's Library as
//! `organization/model-name-astro-atlas-streaming` unless
//! `--output-model-id` names another, and the source is never changed. The
//! verb reads and writes only the disk: no daemon, no MLX. Progress goes to
//! standard error every five seconds with a percentage and an ETA; standard
//! output carries only the final summary.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use astronomical_expert_atlas::{
    AtlasConversionError, ConversionProgress, ConversionProgressObserver, ConversionRequest,
    ConversionStage, atlas_model_id_for, convert_to_atlas, organization_and_name,
};

/// Arguments after `astronomical models convert-atlas`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ConvertAtlasArguments {
    pub source_model: String,
    pub output_model_id: Option<String>,
}

/// Collaborators the verb needs, injected so tests can point it at a
/// scratch Library and capture what it prints.
pub struct ConvertAtlasDependencies<'output> {
    /// This instance's Library folder (`.../models`).
    pub models_directory: PathBuf,
    pub stdout: &'output mut dyn Write,
    pub stderr: &'output mut dyn Write,
    pub progress_interval: Duration,
}

#[derive(Debug, thiserror::Error)]
pub enum ConvertAtlasError {
    #[error(
        "no model {source_model:?}: it is neither a folder nor a model in the Library at {models_directory}"
    )]
    SourceNotFound {
        source_model: String,
        models_directory: String,
    },
    #[error("{0}")]
    InvalidModelId(#[from] astronomical_expert_atlas::atlas_model_naming::ModelIdError),
    #[error(transparent)]
    Conversion(#[from] AtlasConversionError),
    #[error("could not write the conversion summary: {0}")]
    SummaryUnwritable(std::io::Error),
}

/// The source folder and the id the Atlas will be named after.
#[derive(Debug, PartialEq, Eq)]
pub struct ResolvedSource {
    pub source_directory: PathBuf,
    pub source_model_id: String,
}

/// Parses the arguments after `models convert-atlas`.
pub fn parse_convert_atlas_arguments(
    trailing_arguments: &[std::ffi::OsString],
) -> Result<ConvertAtlasArguments, crate::UsageError> {
    let mut source_model = None;
    let mut output_model_id = None;
    let mut remaining_arguments = trailing_arguments.iter();
    while let Some(argument) = remaining_arguments.next() {
        let argument_text = argument.to_string_lossy().into_owned();
        if argument_text == "--output-model-id" {
            let output_value = remaining_arguments
                .next()
                .ok_or(crate::UsageError::MissingValue("--output-model-id"))?;
            if output_model_id
                .replace(output_value.to_string_lossy().into_owned())
                .is_some()
            {
                return Err(crate::UsageError::UnknownArgument(argument_text));
            }
        } else if argument_text.starts_with('-') || source_model.is_some() {
            return Err(crate::UsageError::UnknownArgument(argument_text));
        } else {
            source_model = Some(argument_text);
        }
    }
    let source_model = source_model.ok_or(crate::UsageError::ConvertAtlasModelRequired)?;
    Ok(ConvertAtlasArguments {
        source_model,
        output_model_id,
    })
}

pub fn run_convert_atlas(
    convert_arguments: &ConvertAtlasArguments,
    dependencies: ConvertAtlasDependencies<'_>,
) -> Result<(), ConvertAtlasError> {
    let resolved_source = resolve_source(
        &convert_arguments.source_model,
        &dependencies.models_directory,
    )?;
    let output_model_id = convert_arguments
        .output_model_id
        .clone()
        .unwrap_or_else(|| atlas_model_id_for(&resolved_source.source_model_id));
    let (output_organization, output_name) = organization_and_name(&output_model_id)?;
    let output_directory = dependencies
        .models_directory
        .join(output_organization)
        .join(output_name);
    let mut progress_renderer = ProgressRenderer {
        stderr: dependencies.stderr,
        progress_interval: dependencies.progress_interval,
        last_rendered_at: None,
    };
    let conversion_summary = convert_to_atlas(
        &ConversionRequest {
            source_directory: &resolved_source.source_directory,
            source_model_id: &resolved_source.source_model_id,
            output_directory: &output_directory,
        },
        &mut progress_renderer,
    )?;
    writeln!(
        dependencies.stdout,
        "Converted {}\n     into {output_model_id}\n     {:.1} GB, {} expert layers, every expert byte read back and verified\n     at {}",
        resolved_source.source_model_id,
        conversion_summary.atlas_byte_count as f64 / 1e9,
        conversion_summary.expert_layer_count,
        conversion_summary.output_directory.display()
    )
    .map_err(ConvertAtlasError::SummaryUnwritable)
}

/// A folder is taken as-is and named `parent/folder`; anything else must be
/// an `organization/name` id present in the Library.
pub fn resolve_source(
    source_model: &str,
    models_directory: &Path,
) -> Result<ResolvedSource, ConvertAtlasError> {
    let source_path = Path::new(source_model);
    if source_path.is_dir() && source_path.join("config.json").is_file() {
        let source_directory =
            std::fs::canonicalize(source_path).unwrap_or_else(|_| source_path.to_path_buf());
        return Ok(ResolvedSource {
            source_model_id: folder_model_id(&source_directory),
            source_directory,
        });
    }
    let (organization, name) = organization_and_name(source_model)?;
    let library_directory = models_directory.join(organization).join(name);
    if !library_directory.is_dir() {
        return Err(ConvertAtlasError::SourceNotFound {
            source_model: source_model.to_owned(),
            models_directory: models_directory.display().to_string(),
        });
    }
    Ok(ResolvedSource {
        source_directory: library_directory,
        source_model_id: source_model.to_owned(),
    })
}

/// `.../organization/model-name` -> `organization/model-name`, the Library's
/// own naming, so a folder converts to the same id its Library copy would.
fn folder_model_id(source_directory: &Path) -> String {
    let folder_name = |path: Option<&Path>| {
        path.and_then(Path::file_name)
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default()
    };
    format!(
        "{}/{}",
        folder_name(source_directory.parent()),
        folder_name(Some(source_directory))
    )
}

/// Renders conversion progress at most once per interval, always on the
/// final layer, as `[convert-atlas] layer 12/40 (30%) 11.2 of 34.4 GB, ETA 9s`.
struct ProgressRenderer<'output> {
    stderr: &'output mut dyn Write,
    progress_interval: Duration,
    last_rendered_at: Option<Instant>,
}

impl ConversionProgressObserver for ProgressRenderer<'_> {
    fn observe(&mut self, conversion_progress: &ConversionProgress) {
        let is_final_layer = conversion_progress.completed_layer_count
            == conversion_progress.total_layer_count
            && conversion_progress.stage == ConversionStage::ExpertLayers;
        let interval_elapsed = self
            .last_rendered_at
            .is_none_or(|last_rendered_at| last_rendered_at.elapsed() >= self.progress_interval);
        if !is_final_layer && !interval_elapsed {
            return;
        }
        self.last_rendered_at = Some(Instant::now());
        let _unwritable_progress_is_not_fatal =
            writeln!(self.stderr, "{}", progress_line(conversion_progress));
    }
}

fn progress_line(conversion_progress: &ConversionProgress) -> String {
    let completed_fraction = conversion_progress.written_byte_count as f64
        / conversion_progress.total_byte_count.max(1) as f64;
    let elapsed_seconds = conversion_progress.started_at.elapsed().as_secs_f64();
    let eta_seconds = if completed_fraction > 0.0 {
        elapsed_seconds / completed_fraction - elapsed_seconds
    } else {
        f64::NAN
    };
    let stage_description = match conversion_progress.stage {
        ConversionStage::ResidentBundle => "resident weights".to_owned(),
        ConversionStage::ExpertLayers => format!(
            "layer {}/{}",
            conversion_progress.completed_layer_count, conversion_progress.total_layer_count
        ),
    };
    format!(
        "[convert-atlas] {stage_description} ({:.0}%) {:.1} of {:.1} GB, ETA {eta_seconds:.0}s",
        completed_fraction * 100.0,
        conversion_progress.written_byte_count as f64 / 1e9,
        conversion_progress.total_byte_count as f64 / 1e9
    )
}

#[cfg(test)]
#[path = "convert_atlas_tests.rs"]
mod convert_atlas_tests;
