use super::*;

#[test]
fn should_flush_reasoning_and_request_model_visible_correction_for_an_undeclared_function() {
    let declared_tools = [
        ChatToolDefinition {
            name: "open-brain_openbrain_recall".to_owned(),
            description: None,
            parameters_json: r#"{"type":"object","properties":{"query":{"type":"string"}}}"#
                .to_owned(),
        },
        ChatToolDefinition {
            name: "bash".to_owned(),
            description: None,
            parameters_json: r#"{"type":"object","properties":{"command":{"type":"string"}}}"#
                .to_owned(),
        },
    ];
    let mut output_parser =
        Qwen3_5MoEOutputParser::new(&declared_tools).expect("the declared tools should be valid");

    let tool_call_xml = format!(
        "{THINK_START}I should search memory.{THINK_END}{TOOL_CALL_START}\n<function=open_brain>\n<parameter=query>repo history\n</parameter>\n</function>\n{TOOL_CALL_END}"
    );
    let output_events = output_parser.push_fragment(&tool_call_xml).expect(
        "an undeclared function call should request model-visible correction, not an error",
    );

    assert!(
        output_events.contains(&Qwen3_5MoEOutputEvent::ReasoningDelta(
            "I should search memory.".to_owned()
        )),
        "reasoning emitted before the undeclared function should be flushed before the diagnostic"
    );

    assert!(
        !output_events
            .iter()
            .any(|event| matches!(event, Qwen3_5MoEOutputEvent::TextDelta(_))),
        "undeclared function correction should be sent back to the model, not streamed as assistant text"
    );

    let correction_text = output_events
        .iter()
        .find_map(|event| match event {
            Qwen3_5MoEOutputEvent::ModelVisibleCorrection { correction_text } => {
                Some(correction_text.as_str())
            }
            _ => None,
        })
        .expect("the parser should request a model-visible correction");

    assert!(
        correction_text.contains("open_brain"),
        "correction should name the undeclared function the model attempted: got {correction_text}"
    );
    assert!(
        !correction_text.contains("closest declared tool"),
        "correction should not guess a replacement tool: got {correction_text}"
    );
    assert!(
        !correction_text.contains("open-brain_openbrain_recall"),
        "correction should not suggest any declared tool: got {correction_text}"
    );
    assert!(
        !correction_text.contains("bash"),
        "correction should not suggest the misleading shortest declared tool: got {correction_text}"
    );
    assert!(
        correction_text.contains("Please correct the tool call"),
        "correction should explicitly ask the model to correct the tool call: got {correction_text}"
    );
}
