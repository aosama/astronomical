use astronomical_rest_contract::{OpenAiResponse, OpenAiResponseOutputItem, OpenAiResponseUsage};

#[test]
fn should_serialize_raw_model_reasoning_as_a_plaintext_summary_without_encrypted_content() {
    let response = OpenAiResponse::completed(
        "resp_instance-7",
        1_753_000_000,
        1_753_000_007,
        "mlx-community/Ornith-1.0-35B-OptiQ-4bit",
        Some("Be precise.".to_owned()),
        vec![
            OpenAiResponseOutputItem::reasoning("rs_instance-7", "Inspect first."),
            OpenAiResponseOutputItem::message("msg_instance-7", "Done."),
            OpenAiResponseOutputItem::function_call(
                "fc_instance-7-0",
                "call_instance-7-0",
                "read",
                r#"{"filePath":"README.md"}"#,
            ),
        ],
        OpenAiResponseUsage::new(100, 20, 64, 7).expect("usage should not overflow"),
    );

    let response_document = serde_json::to_value(response).expect("the response should serialize");

    assert_eq!(response_document["object"], "response");
    assert_eq!(response_document["status"], "completed");
    assert_eq!(response_document["created_at"], 1_753_000_000);
    assert_eq!(response_document["completed_at"], 1_753_000_007);
    assert_eq!(response_document["output"][0]["type"], "reasoning");
    assert_eq!(
        response_document["output"][0]["summary"][0]["type"],
        "summary_text"
    );
    assert_eq!(
        response_document["output"][0]["summary"][0]["text"],
        "Inspect first."
    );
    assert_eq!(
        response_document["output"][0]["content"],
        serde_json::json!([])
    );
    assert!(response_document["output"][0]["encrypted_content"].is_null());
    assert_eq!(response_document["output"][1]["type"], "message");
    assert_eq!(
        response_document["output"][1]["content"][0]["text"],
        "Done."
    );
    assert_eq!(response_document["output"][2]["type"], "function_call");
    assert_eq!(response_document["output"][2]["name"], "read");
    assert_eq!(response_document["usage"]["input_tokens"], 100);
    assert_eq!(
        response_document["usage"]["input_tokens_details"]["cached_tokens"],
        64
    );
    assert_eq!(
        response_document["usage"]["output_tokens_details"]["reasoning_tokens"],
        7
    );
    assert_eq!(response_document["usage"]["total_tokens"], 120);
}
