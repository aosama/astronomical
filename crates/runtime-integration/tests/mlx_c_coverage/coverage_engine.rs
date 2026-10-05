//! The MLX-C API coverage contract: the pinned upstream public surface and
//! the compiled bridge must agree exactly.
//!
//! The engine in this file is deliberately pure source analysis: it parses
//! the provisioned extraction-tree headers the way bindgen consumed them and
//! compares the result against the bridge inventory the build script emits.
//! No GPU work and no MLX calls run here, so the contract is hermetic and
//! bounded.

use std::collections::BTreeSet;

/// The public upstream surface parsed from the pinned headers.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct HeaderSurface {
    pub functions: BTreeSet<String>,
    pub types: BTreeSet<String>,
}

/// One side of the coverage comparison.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct BridgeInventory {
    pub functions: BTreeSet<String>,
    pub types: BTreeSet<String>,
}

/// Documented exceptions to full coverage; every entry carries a written
/// justification and appears in the inventory.
pub type Exclusions = &'static [(&'static str, &'static str)];

/// Every coverage discrepancy, each naming the offending symbol.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct CoverageGaps {
    pub unbridged_functions: BTreeSet<String>,
    pub unbridged_types: BTreeSet<String>,
    pub stale_functions: BTreeSet<String>,
    pub stale_types: BTreeSet<String>,
    pub unjustified_exclusions: Vec<String>,
}

impl CoverageGaps {
    /// Whether the bridge is complete for the parsed surface.
    pub fn is_empty(&self) -> bool {
        self == &Self::default()
    }

    /// A human-readable report for every gap, sorted for stable output.
    pub fn report(&self) -> String {
        let mut lines = Vec::new();
        for name in &self.unbridged_functions {
            lines.push(format!("unbridged function: {name}"));
        }
        for name in &self.unbridged_types {
            lines.push(format!("unbridged type: {name}"));
        }
        for name in &self.stale_functions {
            lines.push(format!("bridged function missing from headers: {name}"));
        }
        for name in &self.stale_types {
            lines.push(format!("bridged type missing from headers: {name}"));
        }
        for name in &self.unjustified_exclusions {
            lines.push(format!("exclusion without justification: {name}"));
        }
        lines.join("\n")
    }
}

/// Compares one parsed surface against the bridge and the exclusions
/// registry.
pub fn coverage_gaps(
    surface: &HeaderSurface,
    bridged: &BridgeInventory,
    exclusions: Exclusions,
) -> CoverageGaps {
    let excluded_names: BTreeSet<String> = exclusions
        .iter()
        .map(|(name, _reason)| (*name).to_owned())
        .collect();
    let mut gaps = CoverageGaps {
        unbridged_functions: surface
            .functions
            .difference(&bridged.functions)
            .cloned()
            .collect(),
        unbridged_types: surface.types.difference(&bridged.types).cloned().collect(),
        stale_functions: bridged
            .functions
            .difference(&surface.functions)
            .cloned()
            .collect(),
        stale_types: bridged.types.difference(&surface.types).cloned().collect(),
        unjustified_exclusions: Vec::new(),
    };
    gaps.unbridged_functions
        .retain(|name| !excluded_names.contains(name));
    gaps.unbridged_types
        .retain(|name| !excluded_names.contains(name));
    gaps.stale_functions
        .retain(|name| !excluded_names.contains(name));
    gaps.stale_types
        .retain(|name| !excluded_names.contains(name));
    for (name, reason) in exclusions {
        if reason.trim().is_empty() {
            gaps.unjustified_exclusions.push((*name).to_owned());
            continue;
        }
        let names_a_real_symbol = surface.functions.contains(*name)
            || surface.types.contains(*name)
            || bridged.functions.contains(*name)
            || bridged.types.contains(*name);
        if !names_a_real_symbol {
            gaps.unjustified_exclusions.push((*name).to_owned());
        }
    }
    gaps
}

/// Removes C comments so declaration scanning sees only code.
fn strip_comments(header_text: &str) -> String {
    let mut stripped = String::with_capacity(header_text.len());
    let mut characters = header_text.chars().peekable();
    let mut in_block_comment = false;
    while let Some(character) = characters.next() {
        if in_block_comment {
            if character == '*' && characters.peek() == Some(&'/') {
                characters.next();
                in_block_comment = false;
            }
            continue;
        }
        if character == '/' && characters.peek() == Some(&'*') {
            characters.next();
            in_block_comment = true;
            continue;
        }
        if character == '/' && characters.peek() == Some(&'/') {
            for skipped in characters.by_ref() {
                if skipped == '\n' {
                    stripped.push('\n');
                    break;
                }
            }
            continue;
        }
        stripped.push(character);
    }
    stripped
}

/// Removes preprocessor directives (`#ifdef`, `#endif`, `#define`, …) so
/// declaration scanning never mistakes them for return types; conditional
/// blocks collapse to their contents, matching what the provisioned bindgen
/// extraction saw.
fn strip_preprocessor_directives(stripped: &str) -> String {
    stripped
        .lines()
        .filter(|line| !line.trim_start().starts_with('#'))
        .collect::<Vec<_>>()
        .join("\n")
}

fn identifier_boundary_before(text: &[char], position: usize) -> bool {
    position == 0 || !text[position - 1].is_ascii_alphanumeric() && text[position - 1] != '_'
}

fn read_identifier(text: &[char], start: usize) -> (String, usize) {
    let mut end = start;
    while end < text.len() && (text[end].is_ascii_alphanumeric() || text[end] == '_') {
        end += 1;
    }
    (text[start..end].iter().collect(), end)
}

/// Collects every `mlx_*` identifier in the text.
fn mlx_identifiers(stripped: &str) -> Vec<(String, usize)> {
    let characters: Vec<char> = stripped.chars().collect();
    let mut identifiers = Vec::new();
    let mut position = 0;
    while position < characters.len() {
        if characters[position] == 'm' && identifier_boundary_before(&characters, position) {
            let (identifier, end) = read_identifier(&characters, position);
            if identifier.starts_with("mlx_") {
                identifiers.push((identifier, position));
                position = end;
                continue;
            }
            position = end.max(position + 1);
            continue;
        }
        position += 1;
    }
    identifiers
}

/// The statement boundaries (`;`, `{`, `}`) in the stripped text.
fn statement_boundaries(stripped: &str) -> Vec<(usize, char)> {
    stripped
        .char_indices()
        .filter(|(_index, character)| matches!(character, ';' | '{' | '}'))
        .collect()
}

/// Parses one header's text into its public function and type names.
pub fn parse_header_surface(header_text: &str) -> HeaderSurface {
    let stripped = strip_preprocessor_directives(&strip_comments(header_text));
    let characters: Vec<char> = stripped.chars().collect();
    let boundaries = statement_boundaries(&stripped);
    let mut surface = HeaderSurface::default();

    for (identifier, position) in mlx_identifiers(&stripped) {
        let after_identifier = position + identifier.chars().count();
        let rest: String = characters[after_identifier..]
            .iter()
            .take_while(|character| character.is_whitespace())
            .collect();
        let next_is_open_paren = characters
            .get(after_identifier + rest.chars().count())
            .is_some_and(|character| *character == '(');
        if !next_is_open_paren {
            continue;
        }
        if !is_function_declaration(&characters, after_identifier) {
            continue;
        }
        let statement_start = boundaries
            .iter()
            .rev()
            .find(|(boundary, _)| *boundary < position)
            .map(|(boundary, _)| boundary + 1)
            .unwrap_or(0);
        let return_text: String = characters[statement_start..position]
            .iter()
            .collect::<String>()
            .trim()
            .to_owned();
        if return_text.is_empty()
            || return_text.contains('(')
            || return_text.contains('#')
            || !return_text
                .chars()
                .last()
                .is_some_and(|character| character.is_ascii_alphanumeric() || character == '*')
        {
            continue;
        }
        surface.functions.insert(identifier);
    }

    for typedef_statement in typedef_statements(&stripped) {
        for (identifier, _position) in mlx_identifiers(&typedef_statement) {
            surface.types.insert(identifier);
        }
    }
    surface
}

/// Whether the `(` after the identifier closes into a `;`-terminated
/// declaration.
fn is_function_declaration(characters: &[char], open_paren_position: usize) -> bool {
    let mut depth = 0_usize;
    let mut position = open_paren_position;
    while position < characters.len() {
        match characters[position] {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    let mut tail = position + 1;
                    while tail < characters.len() && characters[tail].is_whitespace() {
                        tail += 1;
                    }
                    return characters.get(tail) == Some(&';');
                }
            }
            _ => {}
        }
        position += 1;
    }
    false
}

/// Every `;`-terminated statement that declares a type.
///
/// Struct bodies contain `;` after each member, so a
/// `typedef struct mlx_stream_ { … } mlx_stream;` declaration splits into a
/// head fragment carrying the tag and a `}`-leading closer fragment carrying
/// the alias; both fragments declare public type names.
fn typedef_statements(stripped: &str) -> Vec<String> {
    let mut statements = Vec::new();
    let mut statement_start = 0_usize;
    for (index, character) in stripped.char_indices() {
        if character == ';' {
            let statement = &stripped[statement_start..index];
            if statement.contains("typedef")
                || statement.contains("struct mlx_")
                || statement.trim_start().starts_with('}')
            {
                statements.push(statement.to_owned());
            }
            statement_start = index + 1;
        }
    }
    statements
}
