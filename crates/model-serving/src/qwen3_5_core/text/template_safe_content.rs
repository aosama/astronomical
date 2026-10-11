//! Escapes untrusted text that shares a Qwen prompt with reserved control markers.

/// Preserves ordinary less-than signs while neutralizing every reserved Qwen marker prefix.
pub(super) fn append_template_safe_content(rendered_prompt: &mut String, untrusted_content: &str) {
    let mut remaining_content = untrusted_content;
    while let Some(marker_offset) = remaining_content.find('<') {
        rendered_prompt.push_str(&remaining_content[..marker_offset]);
        let marker_suffix = &remaining_content[marker_offset + 1..];
        if starts_reserved_template_marker(marker_suffix) {
            rendered_prompt.push_str("&lt;");
        } else {
            rendered_prompt.push('<');
        }
        remaining_content = marker_suffix;
    }
    rendered_prompt.push_str(remaining_content);
}

fn starts_reserved_template_marker(marker_suffix: &str) -> bool {
    marker_suffix.starts_with('|')
        || marker_suffix.starts_with("think>")
        || marker_suffix.starts_with("/think>")
        || marker_suffix.starts_with("tool_call>")
        || marker_suffix.starts_with("/tool_call>")
        || marker_suffix.starts_with("tool_response>")
        || marker_suffix.starts_with("/tool_response>")
        || marker_suffix.starts_with("tools>")
        || marker_suffix.starts_with("/tools>")
        || marker_suffix.starts_with("function=")
        || marker_suffix.starts_with("/function>")
        || marker_suffix.starts_with("parameter=")
        || marker_suffix.starts_with("/parameter>")
}
