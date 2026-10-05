import XCTest;
import RestContract;
import IpcProtocol;

/// Ported from crates/rest-contract/tests/rest_api/openai_chat_completion_request/image_content.rs.
final class ChatCompletionRequestImageContentTests: XCTestCase {

    func testShouldRejectAFileImageUrlBeforeWorkerAdmission() throws {
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
            XCTFail("file image URLs must be rejected before worker admission");
            return;
        } catch let validationError as OpenAiChatCompletionValidationError {
            XCTAssertEqual(validationError, .unsupportedImageUrlScheme);
        }
    }

    func testShouldAcceptADataUriImageContentPart() throws {
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
            XCTFail("expected a user message with images");
            return;
        }
        XCTAssertEqual(content, "What is in this picture?");
        XCTAssertEqual(images.count, 1, "exactly one image should be decoded");
        let image: ImageInput.OpenAiImageInput = images[0];
        XCTAssertTrue(image.decodedBytes().count > 50,
            "the decoded image bytes should be the raw PNG payload");
        XCTAssertEqual(image.mimeType(), "image/png");
    }

    func testShouldRejectAnHttpImageUrl() throws {
        try Self.assertUnsupportedScheme(#"https://example.com/image.png"#,
            because: "http image URLs must be rejected to preserve the local-only privacy model");
    }

    func testShouldRejectAFileImageUrl() throws {
        try Self.assertUnsupportedScheme(#"file:///tmp/image.png"#,
            because: "file image URLs must be rejected to avoid local-path attack surface");
    }

    func testShouldRejectANonImageDataUriMimeType() throws {
        let chatCompletionRequest: OpenAiChatCompletionRequest = try Self.requestWithImageUrl(
            "data:text/plain;base64,SGVsbG8=");
        do {
            try chatCompletionRequest.validate();
            XCTFail("non-image MIME types must be rejected");
            return;
        } catch let validationError as OpenAiChatCompletionValidationError {
            guard case .unsupportedImageMimeType = validationError else {
                XCTFail("expected UnsupportedImageMimeType, got \(validationError)");
                return;
            }
        }
    }

    func testShouldRejectAMalformedDataUriWithoutAComma() throws {
        let chatCompletionRequest: OpenAiChatCompletionRequest = try Self.requestWithImageUrl(
            "data:image/png;base64NOCOMMA");
        do {
            try chatCompletionRequest.validate();
            XCTFail("a data URI without a comma separator must be rejected");
            return;
        } catch let validationError as OpenAiChatCompletionValidationError {
            XCTAssertEqual(validationError, .malformedDataUri);
        }
    }

    func testShouldRejectInvalidBase64InADataUri() throws {
        let chatCompletionRequest: OpenAiChatCompletionRequest = try Self.requestWithImageUrl(
            "data:image/png;base64,!!!not-valid-base64!!!");
        do {
            try chatCompletionRequest.validate();
            XCTFail("invalid base64 payload must be rejected");
            return;
        } catch let validationError as OpenAiChatCompletionValidationError {
            XCTAssertEqual(validationError, .invalidBase64);
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
            XCTFail(reason);
            return;
        } catch let validationError as OpenAiChatCompletionValidationError {
            guard case .unsupportedImageUrlScheme = validationError else {
                XCTFail("expected UnsupportedImageUrlScheme, got \(validationError)");
                return;
            }
        }
    }
}
