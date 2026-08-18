use astronomical_model_serving::{LagunaPromptRenderer, LagunaTextArtifactError};
use serde_json::json;

use super::text_support::{
    ASSISTANT_END_TOKEN_ID, DEFAULT_SYSTEM_MESSAGE, GENERATION_ONLY_EOS_TOKEN_ID,
    MODEL_EOS_TOKEN_ID, POOLSIDE_TEMPLATE, ROMEO_AND_JULIET_SOURCE, SyntheticLagunaTextArtifact,
    romeo_and_juliet_command,
};

const FICTIONAL_ARTIFACT_SYSTEM_MESSAGE: &str =
    "You are the fictional Verona stage archivist; answer only from the supplied play.";
const EXCESSIVE_TEMPLATE_INCLUDE_DEPTH: usize = 9;

#[test]
fn should_normalize_complete_end_token_membership_without_a_family_constant() {
    let text_descriptor = SyntheticLagunaTextArtifact::extra_small_inline().normalize();

    // The third ID exists only in generation_config.json, proving normalization takes the union.
    assert_eq!(
        text_descriptor.end_token_ids(),
        &[
            MODEL_EOS_TOKEN_ID,
            ASSISTANT_END_TOKEN_ID,
            GENERATION_ONLY_EOS_TOKEN_ID,
        ]
    );
    assert!(text_descriptor.is_end_token(MODEL_EOS_TOKEN_ID));
    assert!(text_descriptor.is_end_token(ASSISTANT_END_TOKEN_ID));
    assert!(text_descriptor.is_end_token(GENERATION_ONLY_EOS_TOKEN_ID));
}

#[test]
fn should_normalize_inline_and_nested_included_poolside_templates_equivalently() {
    let inline_descriptor = SyntheticLagunaTextArtifact::extra_small_inline().normalize();
    let mut included_artifact = SyntheticLagunaTextArtifact::extra_small_inline();
    included_artifact.set_embedded_chat_template("{% include 'wrapper.jinja' %}");
    included_artifact.included_templates.insert(
        "wrapper.jinja".to_owned(),
        b"{% include 'canonical_poolside.jinja' %}".to_vec(),
    );
    included_artifact.included_templates.insert(
        "canonical_poolside.jinja".to_owned(),
        POOLSIDE_TEMPLATE.as_bytes().to_vec(),
    );

    let included_descriptor = included_artifact.normalize();
    let chat_command = romeo_and_juliet_command(9_800, Some(0));
    let inline_prompt = LagunaPromptRenderer::new(&inline_descriptor)
        .render(&chat_command.messages, &chat_command.tools, false)
        .expect("the inline Poolside template should render");
    let included_prompt = LagunaPromptRenderer::new(&included_descriptor)
        .render(&chat_command.messages, &chat_command.tools, false)
        .expect("the included Poolside template should render");

    // Equivalent sources must produce the same model-visible journey, not merely equal bytes.
    assert_eq!(included_prompt, inline_prompt);
    assert_eq!(
        included_descriptor.default_thinking_enabled(),
        inline_descriptor.default_thinking_enabled()
    );
    assert_eq!(
        included_descriptor.preserves_prior_reasoning(),
        inline_descriptor.preserves_prior_reasoning()
    );
}

#[test]
fn should_render_an_exact_representative_poolside_prompt() {
    let text_descriptor = SyntheticLagunaTextArtifact::extra_small_inline().normalize();
    let chat_command = romeo_and_juliet_command(9_818, None);

    let rendered_prompt = LagunaPromptRenderer::new(&text_descriptor)
        .render(&chat_command.messages, &chat_command.tools, true)
        .expect("the representative Poolside prompt should render");
    let expected_tool_definitions = concat!(
        "{\"function\":{\"description\":\"Find one character in the supplied play.\",\"name\":\"find_character\",\"parameters\":{\"properties\":{\"name\":{\"type\":\"string\"}},\"required\":[\"name\"],\"type\":\"object\"}},\"type\":\"function\"}\n",
        "{\"function\":{\"description\":\"Summarize one scene from the supplied play.\",\"name\":\"summarize_scene\",\"parameters\":{\"properties\":{\"scene\":{\"type\":\"string\"}},\"required\":[\"scene\"],\"type\":\"object\"}},\"type\":\"function\"}\n",
    );
    let expected_prompt = format!(
        "\u{3008}|EOS|\u{3009}<system>{DEFAULT_SYSTEM_MESSAGE}\n\n<available_tools>\n\
         {expected_tool_definitions}</available_tools></system>\n\
         <user>{ROMEO_AND_JULIET_SOURCE}</user>\n<assistant><think>"
    );

    // This exact protocol example catches semantic marker or field drift in one readable case.
    assert_eq!(rendered_prompt, expected_prompt);
}

#[test]
fn should_render_the_semantically_valid_artifact_provided_system_message() {
    let mut text_artifact = SyntheticLagunaTextArtifact::extra_small_inline();
    let artifact_template =
        POOLSIDE_TEMPLATE.replace(DEFAULT_SYSTEM_MESSAGE, FICTIONAL_ARTIFACT_SYSTEM_MESSAGE);
    // The old sentence is absent, so normalization cannot pass through allowlisting or a comment.
    assert!(!artifact_template.contains(DEFAULT_SYSTEM_MESSAGE));
    text_artifact.set_embedded_chat_template(artifact_template);
    let text_descriptor = text_artifact.normalize();
    let mut chat_command = romeo_and_juliet_command(9_819, Some(0));
    chat_command.tools.clear();

    let rendered_prompt = LagunaPromptRenderer::new(&text_descriptor)
        .render(&chat_command.messages, &chat_command.tools, false)
        .expect("the semantically valid artifact template should render its own default");

    assert!(rendered_prompt.contains(FICTIONAL_ARTIFACT_SYSTEM_MESSAGE));
    assert!(!rendered_prompt.contains(DEFAULT_SYSTEM_MESSAGE));
    assert!(rendered_prompt.contains(ROMEO_AND_JULIET_SOURCE));
    assert!(rendered_prompt.ends_with("<assistant></think>"));
}

#[test]
fn should_reject_a_missing_artifact_local_template_include() {
    let mut text_artifact = SyntheticLagunaTextArtifact::extra_small_inline();
    text_artifact.set_embedded_chat_template("{% include 'chat_template.jinja' %}");

    let normalization_error = text_artifact
        .try_normalize()
        .expect_err("an unresolved template include must fail before prompt rendering");

    assert!(matches!(
        normalization_error,
        LagunaTextArtifactError::MissingTemplateInclude { include_name }
            if include_name == "chat_template.jinja"
    ));
}

#[test]
fn should_reject_an_artifact_local_template_include_cycle() {
    let mut text_artifact = SyntheticLagunaTextArtifact::extra_small_inline();
    text_artifact.set_embedded_chat_template("{% include 'chat_template.jinja' %}");
    text_artifact.included_templates.insert(
        "chat_template.jinja".to_owned(),
        b"{% include 'cycle.jinja' %}".to_vec(),
    );
    text_artifact.included_templates.insert(
        "cycle.jinja".to_owned(),
        b"{% include 'chat_template.jinja' %}".to_vec(),
    );

    let normalization_error = text_artifact
        .try_normalize()
        .expect_err("cyclic template ownership must be rejected deterministically");

    assert!(matches!(
        normalization_error,
        LagunaTextArtifactError::TemplateIncludeCycle { .. }
    ));
}

#[test]
fn should_reject_an_include_chain_beyond_the_bounded_public_depth() {
    let mut text_artifact = SyntheticLagunaTextArtifact::extra_small_inline();
    text_artifact.set_embedded_chat_template("{% include 'depth_0.jinja' %}");
    for include_depth in 0..EXCESSIVE_TEMPLATE_INCLUDE_DEPTH {
        let include_name = format!("depth_{include_depth}.jinja");
        let include_contents = if include_depth + 1 == EXCESSIVE_TEMPLATE_INCLUDE_DEPTH {
            POOLSIDE_TEMPLATE.to_owned()
        } else {
            format!("{{% include 'depth_{}.jinja' %}}", include_depth + 1)
        };
        text_artifact
            .included_templates
            .insert(include_name, include_contents.into_bytes());
    }

    let normalization_error = text_artifact
        .try_normalize()
        .expect_err("a nine-level include chain must exceed the bounded public depth");

    assert!(matches!(
        normalization_error,
        LagunaTextArtifactError::TemplateIncludeDepthExceeded { .. }
    ));
}

#[test]
fn should_reject_template_include_path_traversal() {
    let mut text_artifact = SyntheticLagunaTextArtifact::extra_small_inline();
    text_artifact.set_embedded_chat_template("{% include '../chat_template.jinja' %}");

    let normalization_error = text_artifact
        .try_normalize()
        .expect_err("a template include must remain in the artifact root");

    assert!(matches!(
        normalization_error,
        LagunaTextArtifactError::TemplateIncludeTraversal { include_name }
            if include_name == "../chat_template.jinja"
    ));
}

#[test]
fn should_reject_ambiguous_inline_and_standalone_template_sources() {
    let mut text_artifact = SyntheticLagunaTextArtifact::extra_small_inline();
    text_artifact.included_templates.insert(
        "chat_template.jinja".to_owned(),
        POOLSIDE_TEMPLATE.as_bytes().to_vec(),
    );

    let normalization_error = text_artifact
        .try_normalize()
        .expect_err("two independently authoritative templates must not be guessed between");

    assert!(matches!(
        normalization_error,
        LagunaTextArtifactError::AmbiguousTemplateSource { .. }
    ));
}

#[test]
fn should_accept_only_matching_poolside_v1_reasoning_and_tool_parser_ids() {
    let supported_descriptor = SyntheticLagunaTextArtifact::extra_small_inline().normalize();

    assert_eq!(supported_descriptor.reasoning_parser_id(), "poolside_v1");
    assert_eq!(supported_descriptor.tool_call_parser_id(), "poolside_v1");

    for parser_field_name in ["reasoning_parser", "tool_call_parser"] {
        let mut unsupported_artifact = SyntheticLagunaTextArtifact::extra_small_inline();
        unsupported_artifact
            .generation_config
            .as_mut()
            .expect("the fixture has generation configuration")[parser_field_name] =
            json!("another_parser");

        let normalization_error = unsupported_artifact
            .try_normalize()
            .expect_err("an unsupported parser must fail before generation");

        assert!(matches!(
            normalization_error,
            LagunaTextArtifactError::UnsupportedParserId {
                field_name,
                parser_id,
            } if field_name == parser_field_name && parser_id == "another_parser"
        ));
    }
}

#[test]
fn should_reject_each_missing_poolside_v1_parser_id() {
    for parser_field_name in ["reasoning_parser", "tool_call_parser"] {
        let mut text_artifact = SyntheticLagunaTextArtifact::extra_small_inline();
        text_artifact
            .generation_config
            .as_mut()
            .and_then(|generation_config| generation_config.as_object_mut())
            .expect("the fixture generation config should be an object")
            .remove(parser_field_name);

        let normalization_error = text_artifact
            .try_normalize()
            .expect_err("both parser contracts are required for safe output interpretation");

        assert!(matches!(
            normalization_error,
            LagunaTextArtifactError::MissingParserId { field_name }
                if field_name == parser_field_name
        ));
    }
}

#[test]
fn should_bound_unsupported_parser_identifiers_in_public_errors() {
    let mut text_artifact = SyntheticLagunaTextArtifact::extra_small_inline();
    text_artifact
        .generation_config
        .as_mut()
        .expect("the fixture has generation configuration")["reasoning_parser"] =
        json!("unsupported".repeat(4_096));

    let normalization_error = text_artifact
        .try_normalize()
        .expect_err("an unsupported parser identifier must fail before generation");

    assert!(matches!(
        &normalization_error,
        LagunaTextArtifactError::UnsupportedParserId { parser_id, .. }
            if parser_id.chars().count() == 128
    ));
    assert!(normalization_error.to_string().len() < 256);
}

#[test]
fn should_keep_xs_like_and_s_like_sampling_policies_separate() {
    let extra_small_sampler = SyntheticLagunaTextArtifact::extra_small_inline()
        .normalize()
        .sampler_config()
        .clone();
    let small_sampler = SyntheticLagunaTextArtifact::small_included()
        .normalize()
        .sampler_config()
        .clone();

    assert!(extra_small_sampler.uses_sampling());
    assert_eq!(extra_small_sampler.temperature_thousandths(), 1_000);
    assert_eq!(extra_small_sampler.top_p_thousandths(), 1_000);
    assert_eq!(extra_small_sampler.top_k(), None);
    assert_eq!(extra_small_sampler.repetition_penalty_thousandths(), 1_000);

    assert!(small_sampler.uses_sampling());
    assert_eq!(small_sampler.temperature_thousandths(), 1_000);
    assert_eq!(small_sampler.top_p_thousandths(), 1_000);
    assert_eq!(small_sampler.top_k(), Some(20));
    assert_eq!(small_sampler.repetition_penalty_thousandths(), 1_050);
}

#[test]
fn should_reject_sampling_values_that_cannot_be_preserved_in_protocol_thousandths() {
    for (field_name, unrepresentable_value) in [
        ("temperature", 0.1234),
        ("top_p", 0.9999),
        ("min_p", 0.0005),
        ("repetition_penalty", 1.0005),
    ] {
        let mut text_artifact = SyntheticLagunaTextArtifact::extra_small_inline();
        text_artifact
            .generation_config
            .as_mut()
            .expect("the fixture has generation configuration")[field_name] =
            json!(unrepresentable_value);

        let normalization_error = text_artifact
            .try_normalize()
            .expect_err("sampling precision must not be silently rounded");

        assert!(matches!(
            normalization_error,
            LagunaTextArtifactError::InvalidNumericField {
                field_name: rejected_field_name,
            } if rejected_field_name == field_name
        ));
    }
}

#[test]
fn should_reject_tokenizer_special_token_identity_disagreement() {
    let mut text_artifact = SyntheticLagunaTextArtifact::extra_small_inline();
    let assistant_end_token_id_key = ASSISTANT_END_TOKEN_ID.to_string();
    text_artifact.tokenizer_config["added_tokens_decoder"][&assistant_end_token_id_key]["content"] =
        json!("<wrong_assistant_end>");

    let normalization_error = text_artifact
        .try_normalize()
        .expect_err("configured and tokenizer token identities must describe the same token");

    assert!(matches!(
        normalization_error,
        LagunaTextArtifactError::SpecialTokenMismatch {
            configured_token_id: ASSISTANT_END_TOKEN_ID,
            ..
        }
    ));
}

#[test]
fn should_expose_tokenizer_special_tokens_by_identity_instead_of_assumed_ids() {
    let text_descriptor = SyntheticLagunaTextArtifact::extra_small_inline().normalize();

    assert_eq!(
        text_descriptor.token_id_for("</assistant>"),
        Some(ASSISTANT_END_TOKEN_ID)
    );
    assert_eq!(
        text_descriptor.token_id_for("<generation_end>"),
        Some(GENERATION_ONLY_EOS_TOKEN_ID)
    );
    assert_eq!(text_descriptor.token_id_for("<not_declared>"), None);
}
