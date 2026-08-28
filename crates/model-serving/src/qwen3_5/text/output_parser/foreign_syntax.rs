//! Normalize non-canonical tool-call dialects into the Qwen3.5 function grammar.
//!
//! Quantized models mix Claude-style attribute tags into Qwen output. Rewriting
//! those tags here lets the existing function/parameter parser stay fail-open
//! instead of inventing a second grammar.

pub(super) fn normalize_foreign_tool_call_syntax(tool_call_body: &str) -> String {
    let with_canonical_function_open = normalize_invoke_function_open(tool_call_body);
    rewrite_attribute_parameter_opens(&with_canonical_function_open)
}

fn normalize_invoke_function_open(tool_call_body: &str) -> String {
    let leading_whitespace_bytes = tool_call_body.len() - tool_call_body.trim_start().len();
    let trimmed = &tool_call_body[leading_whitespace_bytes..];
    let Some(after_invoke_open) = strip_invoke_open(trimmed) else {
        return tool_call_body.to_owned();
    };
    let Some((function_name, after_opening_tag)) = extract_named_attribute(after_invoke_open)
    else {
        return tool_call_body.to_owned();
    };
    if function_name.is_empty() {
        return tool_call_body.to_owned();
    }
    let mut normalized = String::with_capacity(tool_call_body.len());
    normalized.push_str(&tool_call_body[..leading_whitespace_bytes]);
    normalized.push_str(concat!("<", "function="));
    normalized.push_str(function_name);
    normalized.push('>');
    normalized.push_str(after_opening_tag);
    normalized
}

fn strip_invoke_open(trimmed_body: &str) -> Option<&str> {
    trimmed_body
        .strip_prefix(concat!("<", "invoke"))
        .or_else(|| trimmed_body.strip_prefix("invoke"))
}

fn extract_named_attribute(after_tag_name: &str) -> Option<(&str, &str)> {
    let name_keyword_offset = after_tag_name.find("name=")?;
    let after_name_keyword = &after_tag_name[name_keyword_offset + "name=".len()..];
    let quote = after_name_keyword
        .chars()
        .next()
        .filter(|character| matches!(character, '"' | '\''))?;
    let name_end = after_name_keyword[1..].find(quote)?;
    let attribute_name = &after_name_keyword[1..1 + name_end];
    let after_quoted_name = &after_name_keyword[1 + name_end + 1..];
    let after_opening_tag = after_quoted_name
        .strip_prefix('>')
        .unwrap_or(after_quoted_name);
    Some((attribute_name, after_opening_tag))
}

fn rewrite_attribute_parameter_opens(tool_call_body: &str) -> String {
    let mut normalized = String::with_capacity(tool_call_body.len());
    let mut remaining = tool_call_body;
    while let Some(parameter_open_offset) = next_attribute_parameter_open_offset(remaining) {
        normalized.push_str(&remaining[..parameter_open_offset]);
        let after_open = &remaining[parameter_open_offset..];
        let (parameter_name, after_opening_tag) = match consume_attribute_parameter_open(after_open)
        {
            Some(rewritten) => rewritten,
            None => {
                normalized.push(remaining.as_bytes()[parameter_open_offset] as char);
                remaining = &remaining[parameter_open_offset + 1..];
                continue;
            }
        };
        normalized.push_str(concat!("<", "parameter="));
        normalized.push_str(parameter_name);
        normalized.push('>');
        remaining = after_opening_tag;
    }
    normalized.push_str(remaining);
    normalized
}

fn next_attribute_parameter_open_offset(remaining: &str) -> Option<usize> {
    let mut search_from = 0usize;
    while search_from < remaining.len() {
        let haystack = &remaining[search_from..];
        let relative = haystack.find("parameter")?;
        let absolute = search_from + relative;
        let has_opening_bracket = absolute > 0 && remaining.as_bytes()[absolute - 1] == b'<';
        let marker_start = if has_opening_bracket {
            absolute - 1
        } else {
            absolute
        };
        let after_marker = &remaining[absolute + "parameter".len()..];
        let after_whitespace = after_marker.trim_start();
        if after_whitespace.starts_with("name=") {
            return Some(marker_start);
        }
        search_from = absolute + 1;
    }
    None
}

fn consume_attribute_parameter_open(after_open: &str) -> Option<(&str, &str)> {
    let after_optional_bracket = after_open.strip_prefix('<').unwrap_or(after_open);
    let after_parameter = after_optional_bracket.strip_prefix("parameter")?;
    let after_whitespace = after_parameter.trim_start();
    let after_name_keyword = after_whitespace.strip_prefix("name=")?;
    extract_named_attribute_value(after_name_keyword)
}

fn extract_named_attribute_value(after_name_keyword: &str) -> Option<(&str, &str)> {
    let quote = after_name_keyword
        .chars()
        .next()
        .filter(|character| matches!(character, '"' | '\''))?;
    let name_end = after_name_keyword[1..].find(quote)?;
    let parameter_name = &after_name_keyword[1..1 + name_end];
    let after_quoted_name = &after_name_keyword[1 + name_end + 1..];
    let after_opening_tag = after_quoted_name
        .strip_prefix('>')
        .unwrap_or(after_quoted_name);
    Some((parameter_name, after_opening_tag))
}

#[cfg(test)]
mod foreign_syntax_unit_tests {
    use super::normalize_foreign_tool_call_syntax;

    #[test]
    fn should_rewrite_invoke_and_named_parameter_tags() {
        let foreign = concat!(
            "<",
            "invoke name=\"find_character\">",
            "<",
            "parameter name=\"name\">Romeo",
            "<",
            "/parameter>",
        );
        let normalized = normalize_foreign_tool_call_syntax(foreign);
        assert!(normalized.starts_with(concat!("<", "function=find_character>")));
        assert!(normalized.contains(concat!("<", "parameter=name>Romeo")));
    }
}
