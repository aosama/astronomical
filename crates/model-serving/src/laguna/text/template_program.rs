use std::io::{self, Write};

use minijinja::value::{Value, ValueKind, from_args};
use minijinja::{AutoEscape, Environment, Error, ErrorKind, UndefinedBehavior};
use serde::Serialize;

use super::LagunaTextArtifactError;
use super::artifact_template::LagunaResolvedTemplateSources;

const ROOT_TEMPLATE_NAME: &str = "__astronomical_laguna_chat__";
const MAXIMUM_RENDERED_PROMPT_BYTES: usize = 32 * 1024 * 1024;
const TEMPLATE_RENDER_FUEL: u64 = 5_000_000;
const TEMPLATE_RECURSION_LIMIT: usize = 32;

/// Compiled, immutable template environment shared by every request for one artifact.
#[derive(Clone, Debug)]
pub(super) struct LagunaTemplateProgram {
    environment: Environment<'static>,
}

impl LagunaTemplateProgram {
    pub(super) fn compile(
        sources: LagunaResolvedTemplateSources,
    ) -> Result<Self, LagunaTextArtifactError> {
        let mut environment = Environment::new();
        // Match Hugging Face chat-template block whitespace while preserving explicit +/- controls.
        environment.set_trim_blocks(true);
        environment.set_lstrip_blocks(true);
        environment.set_undefined_behavior(UndefinedBehavior::Strict);
        environment.set_auto_escape_callback(|_template_name| AutoEscape::None);
        environment.set_fuel(Some(TEMPLATE_RENDER_FUEL));
        environment.set_recursion_limit(TEMPLATE_RECURSION_LIMIT);
        environment.set_unknown_method_callback(python_compatible_method);

        // Every source was selected and read through retained artifact descriptors.
        for (template_name, template_source) in sources.included_sources {
            environment
                .add_template_owned(template_name, template_source)
                .map_err(LagunaTextArtifactError::TemplateCompilation)?;
        }
        environment
            .add_template_owned(ROOT_TEMPLATE_NAME.to_owned(), sources.root_source)
            .map_err(LagunaTextArtifactError::TemplateCompilation)?;
        Ok(Self { environment })
    }

    pub(super) fn render<Context: Serialize>(
        &self,
        context: &Context,
    ) -> Result<String, LagunaTemplateProgramError> {
        let template = self
            .environment
            .get_template(ROOT_TEMPLATE_NAME)
            .map_err(LagunaTemplateProgramError::Template)?;
        let mut prompt_writer = BoundedPromptWriter::new(MAXIMUM_RENDERED_PROMPT_BYTES);
        let rendering = template.render_captured_to(context, &mut prompt_writer);
        if prompt_writer.exceeded_limit {
            return Err(LagunaTemplateProgramError::OutputTooLarge {
                maximum_bytes: MAXIMUM_RENDERED_PROMPT_BYTES,
            });
        }
        drop(rendering.map_err(LagunaTemplateProgramError::Template)?);
        String::from_utf8(prompt_writer.rendered_bytes)
            .map_err(LagunaTemplateProgramError::OutputNotUtf8)
    }
}

#[derive(Debug)]
pub(super) enum LagunaTemplateProgramError {
    Template(Error),
    OutputTooLarge { maximum_bytes: usize },
    OutputNotUtf8(std::string::FromUtf8Error),
}

fn python_compatible_method(
    state: &minijinja::State<'_, '_>,
    value: &Value,
    method_name: &str,
    arguments: &[Value],
) -> Result<Value, Error> {
    match (value.kind(), method_name) {
        (ValueKind::String, "strip") => {
            let _: () = from_args(arguments)?;
            value
                .as_str()
                .map(str::trim)
                .map(Value::from)
                .ok_or_else(|| Error::from(ErrorKind::UnknownMethod))
        }
        (ValueKind::String, "rstrip") => {
            let _: () = from_args(arguments)?;
            value
                .as_str()
                .map(str::trim_end)
                .map(Value::from)
                .ok_or_else(|| Error::from(ErrorKind::UnknownMethod))
        }
        (ValueKind::Map, "items") => {
            let _: () = from_args(arguments)?;
            state.apply_filter("items", &[value.clone()])
        }
        _ => Err(Error::from(ErrorKind::UnknownMethod)),
    }
}

/// io::Write owner that refuses to allocate beyond the request transport ceiling.
struct BoundedPromptWriter {
    rendered_bytes: Vec<u8>,
    maximum_bytes: usize,
    exceeded_limit: bool,
}

impl BoundedPromptWriter {
    fn new(maximum_bytes: usize) -> Self {
        Self {
            rendered_bytes: Vec::new(),
            maximum_bytes,
            exceeded_limit: false,
        }
    }
}

impl Write for BoundedPromptWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let next_byte_count = self
            .rendered_bytes
            .len()
            .checked_add(bytes.len())
            .unwrap_or(usize::MAX);
        if next_byte_count > self.maximum_bytes {
            self.exceeded_limit = true;
            return Err(io::Error::other(
                "Laguna rendered prompt exceeds its byte limit",
            ));
        }
        self.rendered_bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
