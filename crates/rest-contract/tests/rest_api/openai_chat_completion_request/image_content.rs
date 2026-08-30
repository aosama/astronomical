use super::*;

#[test]
fn should_reject_a_file_image_url_before_worker_admission() {
    let request_json = r#"
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": "file:///example.png"}}]
            }
        ]
    }
    "#;
    let chat_completion_request = serde_json::from_str::<OpenAiChatCompletionRequest>(request_json)
        .expect(
            "the request boundary should decode typed content before rejecting unsupported input",
        );

    let validation_error = chat_completion_request
        .validate()
        .expect_err("file image URLs must be rejected before worker admission");

    assert_eq!(
        validation_error,
        OpenAiChatCompletionValidationError::UnsupportedImageUrlScheme
    );
}

#[test]
fn should_accept_a_data_uri_image_content_part() {
    // A 1x1 red PNG, base64-encoded as a data URI.
    let red_pixel_png_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/iZk9HQAAAABJRU5ErkJggg==";
    let data_uri = format!("data:image/png;base64,{red_pixel_png_base64}");
    let request_json = json!({
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this picture?"},
                {"type": "image_url", "image_url": {"url": data_uri}}
            ]
        }]
    })
    .to_string();

    let chat_completion_request =
        serde_json::from_str::<OpenAiChatCompletionRequest>(&request_json)
            .expect("a request with a data URI image part should deserialize");

    let request_parts = chat_completion_request
        .into_parts()
        .expect("a valid data URI image should be accepted and decoded to image bytes");

    let user_message = match request_parts.messages.as_slice() {
        [OpenAiChatMessageParts::User { content, images }] => {
            assert_eq!(content, "What is in this picture?");
            images
        }
        other => panic!("expected a user message with images, got {other:?}"),
    };
    assert_eq!(user_message.len(), 1, "exactly one image should be decoded");
    let image = &user_message[0];
    assert!(
        image.decoded_bytes().len() > 50,
        "the decoded image bytes should be the raw PNG payload"
    );
    assert_eq!(image.mime_type(), "image/png");
}

#[test]
fn should_reject_an_http_image_url() {
    let request_json = json!({
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}
            ]
        }]
    })
    .to_string();

    let chat_completion_request =
        serde_json::from_str::<OpenAiChatCompletionRequest>(&request_json)
            .expect("the request should deserialize before validation rejects the URL scheme");

    let validation_error = chat_completion_request
        .validate()
        .expect_err("http image URLs must be rejected to preserve the local-only privacy model");

    assert!(matches!(
        validation_error,
        OpenAiChatCompletionValidationError::UnsupportedImageUrlScheme
    ));
}

#[test]
fn should_reject_a_file_image_url() {
    let request_json = json!({
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": "file:///tmp/image.png"}}
            ]
        }]
    })
    .to_string();

    let chat_completion_request =
        serde_json::from_str::<OpenAiChatCompletionRequest>(&request_json)
            .expect("the request should deserialize before validation rejects the URL scheme");

    let validation_error = chat_completion_request
        .validate()
        .expect_err("file image URLs must be rejected to avoid local-path attack surface");

    assert!(matches!(
        validation_error,
        OpenAiChatCompletionValidationError::UnsupportedImageUrlScheme
    ));
}

#[test]
fn should_reject_a_non_image_data_uri_mime_type() {
    let request_json = json!({
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": "data:text/plain;base64,SGVsbG8="}}
            ]
        }]
    })
    .to_string();

    let chat_completion_request =
        serde_json::from_str::<OpenAiChatCompletionRequest>(&request_json)
            .expect("the request should deserialize before validation rejects the MIME type");

    let validation_error = chat_completion_request
        .validate()
        .expect_err("non-image MIME types must be rejected");

    assert!(matches!(
        validation_error,
        OpenAiChatCompletionValidationError::UnsupportedImageMimeType { .. }
    ));
}

#[test]
fn should_reject_a_malformed_data_uri_without_a_comma() {
    let request_json = json!({
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64NOCOMMA"}}
            ]
        }]
    })
    .to_string();

    let chat_completion_request =
        serde_json::from_str::<OpenAiChatCompletionRequest>(&request_json)
            .expect("the request should deserialize before validation rejects the malformed URI");

    let validation_error = chat_completion_request
        .validate()
        .expect_err("a data URI without a comma separator must be rejected");

    assert!(matches!(
        validation_error,
        OpenAiChatCompletionValidationError::MalformedDataUri
    ));
}

#[test]
fn should_reject_invalid_base64_in_a_data_uri() {
    let request_json = json!({
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,!!!not-valid-base64!!!"}}
            ]
        }]
    })
    .to_string();

    let chat_completion_request =
        serde_json::from_str::<OpenAiChatCompletionRequest>(&request_json)
            .expect("the request should deserialize before validation rejects the base64");

    let validation_error = chat_completion_request
        .validate()
        .expect_err("invalid base64 payload must be rejected");

    assert!(matches!(
        validation_error,
        OpenAiChatCompletionValidationError::InvalidBase64
    ));
}
