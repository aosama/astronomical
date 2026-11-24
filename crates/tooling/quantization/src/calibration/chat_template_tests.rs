use super::ChatTemplate;
use crate::calibration::CalibrationMessage;

fn message(role: &str, content: &str) -> CalibrationMessage {
    CalibrationMessage {
        role: role.to_owned(),
        content: content.to_owned(),
    }
}

fn template(source: &str) -> ChatTemplate {
    ChatTemplate::from_source(source.to_owned(), Vec::new()).expect("the test template parses")
}

const ROLE_MARKER_TEMPLATE: &str = "{%- for message in messages %}\n    {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' }}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n{%- endif %}";

#[test]
fn should_render_role_markers_without_a_generation_prompt() {
    let rendered = template(ROLE_MARKER_TEMPLATE)
        .render(
            &[
                message("user", "But soft, what light?"),
                message("assistant", "It is the east."),
            ],
            None,
        )
        .expect("the conversation should render");

    assert_eq!(
        rendered,
        "<|im_start|>user\nBut soft, what light?<|im_end|>\n<|im_start|>assistant\nIt is the east.<|im_end|>\n"
    );
}

#[test]
fn should_write_tool_definitions_through_python_style_json() {
    let tool_definitions: minijinja::Value =
        serde_json::from_str(r#"[{"name":"read","parameters":{"path":"string"}}]"#)
            .expect("the tool definitions parse");

    let rendered = template("{% for tool in tools %}{{ tool | tojson }}{% endfor %}")
        .render(&[message("user", "hello")], Some(&tool_definitions))
        .expect("the tools should render");

    assert_eq!(
        rendered,
        r#"{"name": "read", "parameters": {"path": "string"}}"#
    );
}

#[test]
fn should_support_the_python_string_methods_templates_call() {
    let source = "{%- set content = messages[0].content -%}\
{{ content.startswith('<think>') }}|{{ content.endswith('done') }}|\
{{ content.split('</think>')[-1].lstrip('\\n') }}|{{ content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].strip('\\n') }}";

    let rendered = template(source)
        .render(
            &[message("assistant", "<think>\nweigh it\n</think>\n\ndone")],
            None,
        )
        .expect("the string methods should resolve");

    // Jinja prints a Python boolean, capitalized.
    assert_eq!(rendered, "True|True|done|weigh it");
}

#[test]
fn should_report_the_templates_own_refusal() {
    let refusal = template("{{ raise_exception('No user query found in messages.') }}")
        .render(&[message("assistant", "only me")], None)
        .expect_err("the template refuses the conversation");

    assert!(refusal.contains("No user query found in messages."));
}

#[test]
fn should_expose_the_special_tokens_to_the_template() {
    let special_tokens = vec![("eos_token".to_owned(), "<|im_end|>".to_owned())];
    let chat_template = ChatTemplate::from_source("{{ eos_token }}".to_owned(), special_tokens)
        .expect("the test template parses");

    let rendered = chat_template
        .render(&[message("user", "hello")], None)
        .expect("the token should render");

    assert_eq!(rendered, "<|im_end|>");
}

#[test]
fn should_refuse_a_template_that_does_not_parse() {
    assert!(ChatTemplate::from_source("{% for %}".to_owned(), Vec::new()).is_none());
}

#[test]
fn should_find_no_template_in_a_directory_without_one() {
    let directory = tempfile::tempdir().expect("a temporary directory should be creatable");

    assert!(ChatTemplate::from_model_directory(directory.path()).is_none());
}

#[test]
fn should_prefer_the_standalone_template_file_over_the_tokenizer_config() {
    let directory = tempfile::tempdir().expect("a temporary directory should be creatable");
    std::fs::write(directory.path().join("chat_template.jinja"), "standalone")
        .expect("the template file should be writable");
    std::fs::write(
        directory.path().join("tokenizer_config.json"),
        r#"{"chat_template": "embedded", "eos_token": {"content": "<eos>"}}"#,
    )
    .expect("the tokenizer config should be writable");

    let rendered = ChatTemplate::from_model_directory(directory.path())
        .expect("the directory has a template")
        .render(&[message("user", "hello")], None)
        .expect("the template should render");

    assert_eq!(rendered, "standalone");
}
