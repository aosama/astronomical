//! The signature block a registered kernel's `.metal` file opens with.
//!
//! MLX writes a custom kernel's function signature at run time from the
//! call's arrays, so the `.metal` file alone does not say what its body is
//! compiled against. The block says it, one `// @key value` line a fact, in
//! the file's leading comment lines:
//!
//! ```text
//! // @header astro_primitive_matrix_units.h
//! // @prologue astro_primitive_shared_reduction_body.metal
//! // @body astro_primitive_sorted_expert_matmul_body.metal
//! // @input X bfloat16_t
//! // @input first_rows int32_t may-be-small
//! // @input row_count int32_t always-small
//! // @output Y bfloat16_t
//! // @template N int
//! // @define SCALE
//! // @compile-with N=512 SCALE=0.5f
//! // @contract direct_mlx::sorted_expert_matmul
//! // @probe sorted_expert_matmul
//! ```
//!
//! The `@` keeps the block apart from ordinary comments, which may say
//! anything. Facts not in a block line (each input's run-time shape) are the
//! launcher's.

/// How MLX passes an input, which follows from the input's size at run time:
/// fewer than 8 elements go in `constant` memory, a 0-dimensional array as a
/// reference. The kernel body must compile for every way it can be passed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputPassing {
    /// Never fewer than 8 elements: always a `device` pointer.
    Device,
    /// Sized by run-time data (an expert count, a row count) and so may
    /// arrive as either; the offline compile tries both.
    MayBeSmall,
    /// Always fewer than 8 elements (a `[1]` count): a `constant` pointer.
    AlwaysSmall,
    /// A 0-dimensional array: a `constant` reference.
    Scalar,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelInput {
    pub name: String,
    /// Metal type names; more than one when the launcher passes arrays of
    /// several dtypes (`bfloat16_t|float16_t`). The offline compile tries each.
    pub dtype_choices: Vec<String>,
    pub passing: InputPassing,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelOutput {
    pub name: String,
    pub dtype: String,
}

/// The kind MLX writes a template parameter as.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemplateKind {
    Int,
    Bool,
    Typename,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TemplateParameter {
    pub name: String,
    pub kind: TemplateKind,
}

/// The template and define values of one offline instantiation.
pub type InstantiationValues = Vec<(String, String)>;

/// Every fact one kernel's signature block declares.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SignatureBlock {
    pub header_files: Vec<String>,
    /// Shared bodies MLX receives ahead of this file's own text: the common
    /// start of kernels that end differently.
    pub prologue_files: Vec<String>,
    /// Shared bodies MLX receives after this file's own text: the common
    /// rest of kernels whose own text is a preamble.
    pub body_files: Vec<String>,
    pub inputs: Vec<KernelInput>,
    pub outputs: Vec<KernelOutput>,
    pub template_parameters: Vec<TemplateParameter>,
    pub define_names: Vec<String>,
    pub instantiations: Vec<InstantiationValues>,
    pub has_atomic_outputs: bool,
    /// Where the kernel's GPU exactness contract is.
    pub contract: Option<String>,
    /// The capability probe that proves the kernel at worker start, or
    /// `none` and why there is none.
    pub probe: Option<String>,
}

/// A signature block line that cannot be read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SignatureBlockError {
    pub line_number: usize,
    pub description: String,
}

impl std::fmt::Display for SignatureBlockError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "line {}: {}", self.line_number, self.description)
    }
}

impl std::error::Error for SignatureBlockError {}

const BLOCK_LINE_PREFIX: &str = "// @";

/// Reads the signature block from a registered kernel file's text.
///
/// # Errors
/// Fails on an unknown key, a malformed value, a duplicate name, a kernel
/// without outputs, or an instantiation that leaves a template parameter or
/// define without a value.
pub fn parse_signature_block(file_text: &str) -> Result<SignatureBlock, SignatureBlockError> {
    let mut block = SignatureBlock::default();
    for (line_index, line) in file_text.lines().enumerate() {
        let trimmed_line = line.trim();
        if !trimmed_line.starts_with("//") {
            break;
        }
        let Some(block_line) = trimmed_line.strip_prefix(BLOCK_LINE_PREFIX) else {
            continue;
        };
        read_block_line(&mut block, block_line).map_err(|description| SignatureBlockError {
            line_number: line_index + 1,
            description,
        })?;
    }
    check_block(&block).map_err(|description| SignatureBlockError {
        line_number: 0,
        description,
    })?;
    Ok(block)
}

fn read_block_line(block: &mut SignatureBlock, block_line: &str) -> Result<(), String> {
    let (key, value) = block_line.split_once(' ').unwrap_or((block_line, ""));
    let value = value.trim();
    match key {
        "header" => block.header_files.extend(words(value)),
        "prologue" => block.prologue_files.extend(words(value)),
        "body" => block.body_files.extend(words(value)),
        "contract" => block.contract = Some(non_empty(value, "contract")?),
        "probe" => block.probe = Some(non_empty(value, "probe")?),
        "atomic-outputs" => block.has_atomic_outputs = true,
        declaration_key => read_declaration_line(block, declaration_key, value)?,
    }
    Ok(())
}

/// The keys that declare what MLX's wrapper is written from: buffers,
/// template parameters, defines and the instantiations that give them values.
fn read_declaration_line(block: &mut SignatureBlock, key: &str, value: &str) -> Result<(), String> {
    match key {
        "input" => block.inputs.push(parse_input(value)?),
        "output" => block.outputs.push(parse_output(value)?),
        "template" => block
            .template_parameters
            .push(parse_template_parameter(value)?),
        "define" => block.define_names.push(single_word(value, "define")?),
        "compile-with" => block.instantiations.push(parse_instantiation(value)?),
        unknown_key => return Err(format!("unknown signature key @{unknown_key}")),
    }
    Ok(())
}

fn words(value: &str) -> Vec<String> {
    value.split_whitespace().map(str::to_owned).collect()
}

fn non_empty(value: &str, key: &str) -> Result<String, String> {
    if value.is_empty() {
        return Err(format!("@{key} needs a value"));
    }
    Ok(value.to_owned())
}

fn single_word(value: &str, key: &str) -> Result<String, String> {
    let value_words = words(value);
    match value_words.as_slice() {
        [word] => Ok(word.clone()),
        _ => Err(format!("@{key} takes exactly one name, got `{value}`")),
    }
}

fn parse_input(value: &str) -> Result<KernelInput, String> {
    let value_words = words(value);
    let (name, dtypes, passing_word) = match value_words.as_slice() {
        [name, dtypes] => (name, dtypes, None),
        [name, dtypes, passing_word] => (name, dtypes, Some(passing_word.as_str())),
        _ => {
            return Err(format!(
                "@input takes `name dtype [passing]`, got `{value}`"
            ));
        }
    };
    let passing = match passing_word {
        None => InputPassing::Device,
        Some("may-be-small") => InputPassing::MayBeSmall,
        Some("always-small") => InputPassing::AlwaysSmall,
        Some("scalar") => InputPassing::Scalar,
        Some(unknown_passing) => return Err(format!("unknown input passing `{unknown_passing}`")),
    };
    Ok(KernelInput {
        name: name.clone(),
        dtype_choices: dtypes.split('|').map(str::to_owned).collect(),
        passing,
    })
}

fn parse_output(value: &str) -> Result<KernelOutput, String> {
    match words(value).as_slice() {
        [name, dtype] => Ok(KernelOutput {
            name: name.clone(),
            dtype: dtype.clone(),
        }),
        _ => Err(format!("@output takes `name dtype`, got `{value}`")),
    }
}

fn parse_template_parameter(value: &str) -> Result<TemplateParameter, String> {
    let value_words = words(value);
    let [name, kind_word] = value_words.as_slice() else {
        return Err(format!("@template takes `name kind`, got `{value}`"));
    };
    let kind = match kind_word.as_str() {
        "int" => TemplateKind::Int,
        "bool" => TemplateKind::Bool,
        "typename" => TemplateKind::Typename,
        unknown_kind => return Err(format!("unknown template kind `{unknown_kind}`")),
    };
    Ok(TemplateParameter {
        name: name.clone(),
        kind,
    })
}

fn parse_instantiation(value: &str) -> Result<InstantiationValues, String> {
    let mut instantiation = InstantiationValues::new();
    for assignment in value.split_whitespace() {
        let Some((name, assigned_value)) = assignment.split_once('=') else {
            return Err(format!(
                "@compile-with takes `NAME=value` pairs, got `{assignment}`"
            ));
        };
        instantiation.push((name.to_owned(), assigned_value.to_owned()));
    }
    Ok(instantiation)
}

fn check_block(block: &SignatureBlock) -> Result<(), String> {
    // MLX's C interface refuses a kernel without an input or an output.
    if block.inputs.is_empty() || block.outputs.is_empty() {
        return Err("a kernel declares at least one @input and one @output".to_owned());
    }
    let mut declared_names: Vec<&str> = Vec::new();
    for input in &block.inputs {
        declared_names.push(&input.name);
    }
    for output in &block.outputs {
        declared_names.push(&output.name);
    }
    check_unique(&declared_names, "buffer")?;
    let parameter_names = instantiated_names(block);
    check_unique(&parameter_names, "template parameter or define")?;
    if !parameter_names.is_empty() && block.instantiations.is_empty() {
        return Err(
            "a kernel with template parameters or defines needs at least one @compile-with"
                .to_owned(),
        );
    }
    for instantiation in &block.instantiations {
        check_instantiation(instantiation, &parameter_names)?;
    }
    Ok(())
}

/// The names each `@compile-with` must give a value: template parameters
/// first, then defines.
pub fn instantiated_names(block: &SignatureBlock) -> Vec<&str> {
    let mut parameter_names: Vec<&str> = Vec::new();
    for template_parameter in &block.template_parameters {
        parameter_names.push(&template_parameter.name);
    }
    for define_name in &block.define_names {
        parameter_names.push(define_name);
    }
    parameter_names
}

fn check_unique(names: &[&str], what: &str) -> Result<(), String> {
    for (name_index, name) in names.iter().enumerate() {
        if names[..name_index].contains(name) {
            return Err(format!("{what} `{name}` is declared twice"));
        }
    }
    Ok(())
}

fn check_instantiation(
    instantiation: &InstantiationValues,
    parameter_names: &[&str],
) -> Result<(), String> {
    for (assigned_name, _) in instantiation {
        if !parameter_names.contains(&assigned_name.as_str()) {
            return Err(format!(
                "@compile-with names `{assigned_name}`, which is not a template parameter or define"
            ));
        }
    }
    for parameter_name in parameter_names {
        let mut assignment_count = 0;
        for (assigned_name, _) in instantiation {
            if assigned_name == parameter_name {
                assignment_count += 1;
            }
        }
        if assignment_count != 1 {
            return Err(format!(
                "@compile-with gives `{parameter_name}` {assignment_count} values; it needs exactly one"
            ));
        }
    }
    Ok(())
}
