use std::collections::{BTreeMap, BTreeSet};
use std::path::{Component, Path};

use serde_json::{Map, Value};

use super::LagunaTextArtifactError;
use super::artifact_documents::{bounded_artifact_text, object_fields, parse_json_document};

pub(crate) const MAXIMUM_TEMPLATE_BYTES: usize = 512 * 1024;
pub(crate) const MAXIMUM_TEMPLATE_SOURCE_COUNT: usize = 16;
pub(crate) const MAXIMUM_TEMPLATE_INCLUDE_DEPTH: usize = 8;
const MAXIMUM_TEMPLATE_INCLUDE_NAME_BYTES: usize = 255;
const SUPPORTED_PARSER_ID: &str = "poolside_v1";

/// Validated, normalized artifact sources ready for one loader-free environment.
#[derive(Debug)]
pub(super) struct LagunaResolvedTemplateSources {
    pub(super) root_source: String,
    pub(super) included_sources: BTreeMap<String, String>,
}

/// Selects the complete static include graph and rejects supplied but unused sources.
pub(super) fn resolve_template_sources(
    tokenizer_config_fields: &Map<String, Value>,
    supplied_included_templates: &BTreeMap<String, Vec<u8>>,
) -> Result<LagunaResolvedTemplateSources, LagunaTextArtifactError> {
    let root_source = root_template(tokenizer_config_fields)?;
    let mut selected_include_names = BTreeSet::new();
    let mut active_include_names = Vec::new();
    let mut maximum_expanded_depth_by_name = BTreeMap::new();
    select_includes(
        root_source,
        supplied_included_templates,
        0,
        &mut active_include_names,
        &mut selected_include_names,
        &mut maximum_expanded_depth_by_name,
    )?;

    if selected_include_names.len() != supplied_included_templates.len() {
        return Err(LagunaTextArtifactError::AmbiguousTemplateSource {
            source_count: supplied_included_templates.len() + 1,
        });
    }
    let total_source_count = selected_include_names.len() + 1;
    if total_source_count > MAXIMUM_TEMPLATE_SOURCE_COUNT {
        return Err(LagunaTextArtifactError::TooManyTemplateSources {
            actual_count: total_source_count,
            maximum_count: MAXIMUM_TEMPLATE_SOURCE_COUNT,
        });
    }

    let mut total_source_bytes = root_source.len();
    let mut included_sources = BTreeMap::new();
    for include_name in selected_include_names {
        let template_bytes = supplied_included_templates
            .get(&include_name)
            .ok_or_else(|| LagunaTextArtifactError::MissingTemplateInclude {
                include_name: include_name.clone(),
            })?;
        total_source_bytes = total_source_bytes
            .checked_add(template_bytes.len())
            .unwrap_or(usize::MAX);
        let template_source = std::str::from_utf8(template_bytes)
            .map_err(LagunaTextArtifactError::TemplateNotUtf8)?;
        included_sources.insert(include_name, normalize_hugging_face_syntax(template_source));
    }
    if total_source_bytes > MAXIMUM_TEMPLATE_BYTES {
        return Err(LagunaTextArtifactError::DocumentTooLarge {
            document_name: "chat template sources",
            actual_bytes: total_source_bytes,
            maximum_bytes: MAXIMUM_TEMPLATE_BYTES,
        });
    }

    Ok(LagunaResolvedTemplateSources {
        root_source: normalize_hugging_face_syntax(root_source),
        included_sources,
    })
}

/// Discovers every root include through the duplicate-aware tokenizer boundary.
pub(crate) fn discover_root_template_includes(
    tokenizer_config_bytes: &[u8],
) -> Result<Vec<String>, LagunaTextArtifactError> {
    let tokenizer_config = parse_json_document("tokenizer config", tokenizer_config_bytes)?;
    let tokenizer_config_fields = object_fields(&tokenizer_config, "tokenizer config")?;
    discover_template_includes(root_template(tokenizer_config_fields)?)
}

/// Discovers only static single-quoted include names from one Jinja source.
pub(crate) fn discover_template_includes(
    template_source: &str,
) -> Result<Vec<String>, LagunaTextArtifactError> {
    let mut include_names = Vec::new();
    let mut remaining_source = template_source;
    while let Some(directive_start) = remaining_source.find("{%") {
        remaining_source = &remaining_source[directive_start + 2..];
        let directive_end = remaining_source.find("%}").ok_or(
            LagunaTextArtifactError::MalformedTemplateContract {
                description: "Jinja block directive is unterminated",
            },
        )?;
        let directive_body = trim_jinja_whitespace_control(&remaining_source[..directive_end]);
        if let Some(include_expression) = directive_body.strip_prefix("include") {
            let include_name = parse_static_include_name(include_expression.trim())?;
            validate_include_name(&include_name)?;
            include_names.push(include_name);
        }
        remaining_source = &remaining_source[directive_end + 2..];
    }
    Ok(include_names)
}

fn root_template(
    tokenizer_config_fields: &Map<String, Value>,
) -> Result<&str, LagunaTextArtifactError> {
    let root_source = tokenizer_config_fields
        .get("chat_template")
        .and_then(Value::as_str)
        .ok_or_else(|| LagunaTextArtifactError::InvalidField {
            field_name: "chat_template".to_owned(),
        })?;
    if root_source.len() > MAXIMUM_TEMPLATE_BYTES {
        return Err(LagunaTextArtifactError::DocumentTooLarge {
            document_name: "chat template",
            actual_bytes: root_source.len(),
            maximum_bytes: MAXIMUM_TEMPLATE_BYTES,
        });
    }
    Ok(root_source)
}

fn select_includes(
    template_source: &str,
    supplied_sources: &BTreeMap<String, Vec<u8>>,
    parent_depth: usize,
    active_include_names: &mut Vec<String>,
    selected_include_names: &mut BTreeSet<String>,
    maximum_expanded_depth_by_name: &mut BTreeMap<String, usize>,
) -> Result<(), LagunaTextArtifactError> {
    for include_name in discover_template_includes(template_source)? {
        let include_depth = parent_depth + 1;
        if include_depth > MAXIMUM_TEMPLATE_INCLUDE_DEPTH {
            return Err(LagunaTextArtifactError::TemplateIncludeDepthExceeded { include_name });
        }
        if active_include_names.contains(&include_name) {
            return Err(LagunaTextArtifactError::TemplateIncludeCycle { include_name });
        }
        let included_bytes = supplied_sources.get(&include_name).ok_or_else(|| {
            LagunaTextArtifactError::MissingTemplateInclude {
                include_name: include_name.clone(),
            }
        })?;
        let included_source = std::str::from_utf8(included_bytes)
            .map_err(LagunaTextArtifactError::TemplateNotUtf8)?;
        let previous_expanded_depth = maximum_expanded_depth_by_name
            .get(&include_name)
            .copied()
            .unwrap_or(0);
        selected_include_names.insert(include_name.clone());
        if selected_include_names.len() + 1 > MAXIMUM_TEMPLATE_SOURCE_COUNT {
            return Err(LagunaTextArtifactError::TooManyTemplateSources {
                actual_count: selected_include_names.len() + 1,
                maximum_count: MAXIMUM_TEMPLATE_SOURCE_COUNT,
            });
        }
        if previous_expanded_depth >= include_depth {
            continue;
        }
        maximum_expanded_depth_by_name.insert(include_name.clone(), include_depth);
        active_include_names.push(include_name.clone());
        select_includes(
            included_source,
            supplied_sources,
            include_depth,
            active_include_names,
            selected_include_names,
            maximum_expanded_depth_by_name,
        )?;
        if active_include_names.pop().is_none() {
            return Err(LagunaTextArtifactError::MalformedTemplateContract {
                description: "include traversal stack became inconsistent",
            });
        }
    }
    Ok(())
}

fn trim_jinja_whitespace_control(directive_source: &str) -> &str {
    let mut directive_body = directive_source.trim();
    if let Some(without_left_control) = directive_body.strip_prefix('-') {
        directive_body = without_left_control.trim();
    }
    if let Some(without_right_control) = directive_body.strip_suffix('-') {
        directive_body = without_right_control.trim();
    }
    directive_body
}

fn parse_static_include_name(include_expression: &str) -> Result<String, LagunaTextArtifactError> {
    let include_name = include_expression
        .strip_prefix('\'')
        .and_then(|quoted_name| quoted_name.strip_suffix('\''))
        .ok_or(LagunaTextArtifactError::MalformedTemplateContract {
            description: "include name must be one static single-quoted path",
        })?;
    if include_name.is_empty() || include_name.contains('\'') {
        return Err(LagunaTextArtifactError::MalformedTemplateContract {
            description: "include name must be one static single-quoted path",
        });
    }
    Ok(include_name.to_owned())
}

fn validate_include_name(include_name: &str) -> Result<(), LagunaTextArtifactError> {
    let include_path = Path::new(include_name);
    let is_artifact_local = !include_name.is_empty()
        && include_name.len() <= MAXIMUM_TEMPLATE_INCLUDE_NAME_BYTES
        && !include_path.is_absolute()
        && !include_name.contains('\\')
        && include_path
            .components()
            .all(|component| matches!(component, Component::Normal(_)));
    if !is_artifact_local {
        return Err(LagunaTextArtifactError::TemplateIncludeTraversal {
            include_name: bounded_artifact_text(include_name),
        });
    }
    Ok(())
}

fn normalize_hugging_face_syntax(template_source: &str) -> String {
    // Hugging Face uses generation blocks only for output annotation. A true
    // block preserves the exact whitespace and rendering semantics in MiniJinja.
    template_source
        .replace("{%- generation -%}", "{%- if true -%}")
        .replace("{%- endgeneration -%}", "{%- endif -%}")
        // MiniJinja JSON is already Unicode-preserving and does not accept this Python keyword.
        .replace("tojson(ensure_ascii=False)", "tojson")
}

pub(super) fn required_parser_id(
    generation_fields: &Map<String, Value>,
    field_name: &str,
) -> Result<String, LagunaTextArtifactError> {
    let parser_id = generation_fields
        .get(field_name)
        .and_then(Value::as_str)
        .ok_or_else(|| LagunaTextArtifactError::MissingParserId {
            field_name: field_name.to_owned(),
        })?;
    if parser_id != SUPPORTED_PARSER_ID {
        return Err(LagunaTextArtifactError::UnsupportedParserId {
            field_name: field_name.to_owned(),
            parser_id: bounded_artifact_text(parser_id),
        });
    }
    Ok(parser_id.to_owned())
}
