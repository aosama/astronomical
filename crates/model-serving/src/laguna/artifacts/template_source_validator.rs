use std::collections::BTreeMap;
use std::path::Path;

use crate::artifact_validation::{
    ArtifactValidationError, RequiredFileProfile, ValidatedRequiredFile,
    read_bounded_required_file_bytes, validate_required_file,
};
use crate::laguna::LagunaTextArtifactError;
use crate::laguna::text::{
    MAXIMUM_TEMPLATE_BYTES, MAXIMUM_TEMPLATE_INCLUDE_DEPTH, MAXIMUM_TEMPLATE_SOURCE_COUNT,
    discover_template_includes,
};

use super::LagunaArtifactValidationError;

/// Retained descriptors and bytes for the complete selected template include graph.
pub(super) struct ValidatedLagunaTemplateSources {
    pub(super) files_by_name: BTreeMap<String, ValidatedRequiredFile>,
    pub(super) bytes_by_name: BTreeMap<String, Vec<u8>>,
}

/// Recursively validates and reads static artifact-local includes without reopening paths.
pub(super) struct LagunaTemplateSourceValidator<'a> {
    model_directory: &'a Path,
}

impl<'a> LagunaTemplateSourceValidator<'a> {
    pub(super) const fn new(model_directory: &'a Path) -> Self {
        Self { model_directory }
    }

    pub(super) fn validate(
        self,
        root_template_source: &str,
        standalone_root_file_name: Option<&str>,
        prevalidated_template_file: Option<(String, ValidatedRequiredFile)>,
    ) -> Result<ValidatedLagunaTemplateSources, LagunaArtifactValidationError> {
        if root_template_source.len() > MAXIMUM_TEMPLATE_BYTES {
            return Err(LagunaTextArtifactError::DocumentTooLarge {
                document_name: "chat template",
                actual_bytes: root_template_source.len(),
                maximum_bytes: MAXIMUM_TEMPLATE_BYTES,
            }
            .into());
        }
        let root_include_names = discover_template_includes(root_template_source)?;
        let root_ancestors = standalone_root_file_name
            .map(|file_name| vec![file_name.to_owned()])
            .unwrap_or_default();
        let mut files_by_name = BTreeMap::new();
        let mut bytes_by_name = BTreeMap::new();
        let mut prevalidated_files_by_name = prevalidated_template_file
            .into_iter()
            .collect::<BTreeMap<_, _>>();
        let mut pending_includes = root_include_names
            .into_iter()
            .map(|include_name| PendingTemplateInclude {
                include_name,
                depth: 1,
                ancestors: root_ancestors.clone(),
            })
            .collect::<Vec<_>>();
        let mut total_template_bytes = root_template_source.len();
        let mut maximum_expanded_depth_by_name = BTreeMap::new();

        while let Some(pending_include) = pending_includes.pop() {
            if pending_include.depth > MAXIMUM_TEMPLATE_INCLUDE_DEPTH {
                return Err(LagunaTextArtifactError::TemplateIncludeDepthExceeded {
                    include_name: pending_include.include_name,
                }
                .into());
            }
            if pending_include
                .ancestors
                .contains(&pending_include.include_name)
            {
                return Err(LagunaTextArtifactError::TemplateIncludeCycle {
                    include_name: pending_include.include_name,
                }
                .into());
            }

            if !bytes_by_name.contains_key(&pending_include.include_name) {
                if files_by_name.len() + 2 > MAXIMUM_TEMPLATE_SOURCE_COUNT {
                    return Err(LagunaTextArtifactError::TooManyTemplateSources {
                        actual_count: files_by_name.len() + 2,
                        maximum_count: MAXIMUM_TEMPLATE_SOURCE_COUNT,
                    }
                    .into());
                }
                let template_file =
                    match prevalidated_files_by_name.remove(&pending_include.include_name) {
                        Some(prevalidated_file) => prevalidated_file,
                        None => self.validate_include_file(&pending_include.include_name)?,
                    };
                let template_bytes = read_bounded_required_file_bytes(
                    &template_file,
                    MAXIMUM_TEMPLATE_BYTES as u64,
                )?;
                total_template_bytes = total_template_bytes
                    .checked_add(template_bytes.len())
                    .unwrap_or(usize::MAX);
                if total_template_bytes > MAXIMUM_TEMPLATE_BYTES {
                    return Err(LagunaTextArtifactError::DocumentTooLarge {
                        document_name: "chat template sources",
                        actual_bytes: total_template_bytes,
                        maximum_bytes: MAXIMUM_TEMPLATE_BYTES,
                    }
                    .into());
                }
                files_by_name.insert(pending_include.include_name.clone(), template_file);
                bytes_by_name.insert(pending_include.include_name.clone(), template_bytes);
            }

            let previous_expanded_depth = maximum_expanded_depth_by_name
                .get(&pending_include.include_name)
                .copied()
                .unwrap_or(0);
            if previous_expanded_depth >= pending_include.depth {
                continue;
            }
            maximum_expanded_depth_by_name
                .insert(pending_include.include_name.clone(), pending_include.depth);

            let template_bytes = bytes_by_name
                .get(&pending_include.include_name)
                .ok_or_else(|| LagunaTextArtifactError::MissingTemplateInclude {
                    include_name: pending_include.include_name.clone(),
                })?;
            let template_source = std::str::from_utf8(template_bytes)
                .map_err(LagunaTextArtifactError::TemplateNotUtf8)?;
            let mut child_ancestors = pending_include.ancestors;
            child_ancestors.push(pending_include.include_name);
            for child_include_name in discover_template_includes(template_source)? {
                pending_includes.push(PendingTemplateInclude {
                    include_name: child_include_name,
                    depth: pending_include.depth + 1,
                    ancestors: child_ancestors.clone(),
                });
            }
        }

        Ok(ValidatedLagunaTemplateSources {
            files_by_name,
            bytes_by_name,
        })
    }

    fn validate_include_file(
        &self,
        include_name: &str,
    ) -> Result<ValidatedRequiredFile, LagunaArtifactValidationError> {
        match validate_required_file(
            self.model_directory,
            &RequiredFileProfile {
                file_name: include_name.to_owned(),
                size_bytes: 0,
            },
        ) {
            Ok(template_file) => Ok(template_file),
            Err(ArtifactValidationError::InspectRequiredFile { source, .. })
                if source.kind() == std::io::ErrorKind::NotFound =>
            {
                Err(LagunaTextArtifactError::MissingTemplateInclude {
                    include_name: include_name.to_owned(),
                }
                .into())
            }
            Err(validation_error) => Err(validation_error.into()),
        }
    }
}

struct PendingTemplateInclude {
    include_name: String,
    depth: usize,
    ancestors: Vec<String>,
}
