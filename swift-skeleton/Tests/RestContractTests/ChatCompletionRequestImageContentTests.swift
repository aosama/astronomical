import Foundation;
import RestContract;
import IpcProtocol;
import Testing;
import JourneyCategories;

/// Ported from crates/rest-contract/tests/rest_api/openai_chat_completion_request/image_content.rs.
@Suite(.tags(.hermeticJourney))
final class ChatCompletionRequestImageContentTests {

    @Test
    func should_reject_a_file_image_url_before_worker_admission() throws {
        let chatCompletionRequest: OpenAiChatCompletionRequest = try OpenAiChatCompletionRequest.decoded(
            wireValue: try RestContractTestFixture.wireValue("""
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": "file:///example.png"}}]
            }
        ]
    }
    """));
        do {
            try chatCompletionRequest.validate();
            Issue.record("file image URLs must be rejected before worker admission");
        } catch let validationError as OpenAiChatCompletionValidationError {
            #expect(validationError == OpenAiChatCompletionValidationError.unsupportedImageUrlScheme);
        }
    }

    @Test
    func should_accept_a_data_uri_image_content_part() throws {
        // A 1x1 red PNG, base64-encoded as a data URI.
        let redPixelPngBase64: String = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/iZk9HQAAAABJRU5ErkJggg==";
        let dataUri: String = "data:image/png;base64,\(redPixelPngBase64)";
        let requestJson: String = """
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this picture?"},
                {"type": "image_url", "image_url": {"url": "\(dataUri)"}}
            ]
        }]
    }
    """;
        let chatCompletionRequest: OpenAiChatCompletionRequest = try OpenAiChatCompletionRequest.decoded(
            wireValue: try RestContractTestFixture.wireValue(requestJson));
        let requestParts: OpenAiChatCompletionRequestParts = try chatCompletionRequest.intoParts();
        guard requestParts.messages.count == 1,
            case .user(let content, let images) = requestParts.messages[0] else {
            Issue.record("expected a user message with images");
            return;
        }
        #expect(content == "What is in this picture?");
        #expect(images.count == 1, "exactly one image should be decoded");
        let image: ImageInput.OpenAiImageInput = images[0];
        #expect(image.decodedBytes().count > 50,
            "the decoded image bytes should be the raw PNG payload");
        #expect(image.mimeType() == "image/png");
    }

    @Test
    func should_reject_an_http_image_url() throws {
        try Self.assertUnsupportedScheme(#"https://example.com/image.png"#,
            because: "http image URLs must be rejected to preserve the local-only privacy model");
    }

    @Test
    func should_reject_a_file_image_url() throws {
        try Self.assertUnsupportedScheme(#"file:///tmp/image.png"#,
            because: "file image URLs must be rejected to avoid local-path attack surface");
    }

    @Test
    func should_reject_a_non_image_data_uri_mime_type() throws {
        let chatCompletionRequest: OpenAiChatCompletionRequest = try Self.requestWithImageUrl(
            "data:text/plain;base64,SGVsbG8=");
        do {
            try chatCompletionRequest.validate();
            Issue.record("non-image MIME types must be rejected");
        } catch let validationError as OpenAiChatCompletionValidationError {
            guard case .unsupportedImageMimeType = validationError else {
                Issue.record("expected UnsupportedImageMimeType, got \(validationError)");
                return;
            }
        }
    }

    @Test
    func should_reject_a_malformed_data_uri_without_a_comma() throws {
        let chatCompletionRequest: OpenAiChatCompletionRequest = try Self.requestWithImageUrl(
            "data:image/png;base64NOCOMMA");
        do {
            try chatCompletionRequest.validate();
            Issue.record("a data URI without a comma separator must be rejected");
        } catch let validationError as OpenAiChatCompletionValidationError {
            #expect(validationError == OpenAiChatCompletionValidationError.malformedDataUri);
        }
    }

    @Test
    func should_reject_invalid_base64_in_a_data_uri() throws {
        let chatCompletionRequest: OpenAiChatCompletionRequest = try Self.requestWithImageUrl(
            "data:image/png;base64,!!!not-valid-base64!!!");
        do {
            try chatCompletionRequest.validate();
            Issue.record("invalid base64 payload must be rejected");
        } catch let validationError as OpenAiChatCompletionValidationError {
            #expect(validationError == OpenAiChatCompletionValidationError.invalidBase64);
        }
    }

    private static func requestWithImageUrl(_ imageUrl: String) throws -> OpenAiChatCompletionRequest {
        let requestJson: String = """
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": "\(imageUrl)"}}
            ]
        }]
    }
    """;
        return try OpenAiChatCompletionRequest.decoded(
            wireValue: try RestContractTestFixture.wireValue(requestJson));
    }

    private static func assertUnsupportedScheme(_ imageUrl: String, because reason: String) throws {
        let chatCompletionRequest: OpenAiChatCompletionRequest = try requestWithImageUrl(imageUrl);
        do {
            try chatCompletionRequest.validate();
            Issue.record(Comment(stringLiteral: reason));
        } catch let validationError as OpenAiChatCompletionValidationError {
            guard case .unsupportedImageUrlScheme = validationError else {
                Issue.record("expected UnsupportedImageUrlScheme, got \(validationError)");
                return;
            }
        }
    }
}
