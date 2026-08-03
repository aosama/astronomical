use super::*;

#[test]
fn should_emit_reasoning_and_text_without_qwen3_5_moe_marker_syntax() {
    let mut output_parser = Qwen3_5MoEOutputParser::new(&[])
        .expect("an empty declared-tool set should be valid for text-only parsing");

    assert_eq!(
        output_parser
            .push_fragment(&format!("{THINK_START}inspect the source"))
            .expect("a complete thinking-start marker should parse"),
        vec![Qwen3_5MoEOutputEvent::ReasoningDelta(
            "inspect the source".to_owned()
        )]
    );
    assert_eq!(
        output_parser
            .push_fragment(&format!("{THINK_END}then edit it."))
            .expect("a complete thinking-end marker should transition into text"),
        vec![Qwen3_5MoEOutputEvent::TextDelta("then edit it.".to_owned())]
    );
    assert!(
        output_parser
            .finish()
            .expect("a completed reasoning/text response should finish")
            .is_empty()
    );
}

#[test]
fn should_continue_reasoning_already_opened_by_the_generation_prompt() {
    let mut output_parser = Qwen3_5MoEOutputParser::new_after_thinking_prefix(&[])
        .expect("an empty declared-tool set should support prompt-opened reasoning");

    assert_eq!(
        output_parser
            .push_fragment(&format!("Inspect first{THINK_END}Then answer."))
            .expect("the prompt-opened reasoning continuation should parse"),
        vec![
            Qwen3_5MoEOutputEvent::ReasoningDelta("Inspect first".to_owned()),
            Qwen3_5MoEOutputEvent::TextDelta("Then answer.".to_owned()),
        ]
    );
}

#[test]
fn should_suppress_late_thinking_blocks_after_visible_text_has_started() {
    let mut output_parser = Qwen3_5MoEOutputParser::new_after_thinking_prefix(&[])
        .expect("an empty declared-tool set should support prompt-opened reasoning");

    assert_eq!(
        output_parser
            .push_fragment(&format!(
                "Initial plan{THINK_END}Visible answer.{THINK_START}late private thought{THINK_END}Done."
            ))
            .expect("late thinking markers should parse as a suppressed control block"),
        vec![
            Qwen3_5MoEOutputEvent::ReasoningDelta("Initial plan".to_owned()),
            Qwen3_5MoEOutputEvent::TextDelta("Visible answer.".to_owned()),
            Qwen3_5MoEOutputEvent::TextDelta("Done.".to_owned()),
        ]
    );
}

#[test]
fn should_finish_with_streamed_reasoning_when_generation_stops_before_the_thinking_end_marker() {
    let mut output_parser = Qwen3_5MoEOutputParser::new_after_thinking_prefix(&[])
        .expect("an empty declared-tool set should support prompt-opened reasoning");

    assert_eq!(
        output_parser
            .push_fragment("Thinking through the short answer")
            .expect("reasoning should stream before the closing marker"),
        vec![Qwen3_5MoEOutputEvent::ReasoningDelta(
            "Thinking through the short answer".to_owned()
        )]
    );
    assert!(
        output_parser
            .finish()
            .expect("stopping inside reasoning should not make the response malformed")
            .is_empty()
    );
}

#[test]
fn should_buffer_control_markers_split_across_token_fragments() {
    let mut output_parser = Qwen3_5MoEOutputParser::new(&[])
        .expect("an empty declared-tool set should be valid for text-only parsing");

    assert!(
        output_parser
            .push_fragment("<thi")
            .expect("a partial thinking marker should remain buffered")
            .is_empty()
    );
    assert_eq!(
        output_parser
            .push_fragment("nk>plan</thi")
            .expect("a partial thinking-end marker should preserve reasoning")
            .as_slice(),
        [Qwen3_5MoEOutputEvent::ReasoningDelta("plan".to_owned())]
    );
    assert_eq!(
        output_parser
            .push_fragment("nk>answer")
            .expect("the completed thinking-end marker should reveal text"),
        vec![Qwen3_5MoEOutputEvent::TextDelta("answer".to_owned())]
    );
}
