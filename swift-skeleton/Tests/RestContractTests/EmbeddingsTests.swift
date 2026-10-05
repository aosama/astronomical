import XCTest;
import RestContract;
import IpcProtocol;

/// Ported from crates/rest-contract/tests/rest_api/openai_embeddings.rs.
final class EmbeddingsTests: XCTestCase {

    func testShouldValidateASingleStringInputIntoOnePart() throws {
        let request: OpenAiEmbeddingsRequest = try OpenAiEmbeddingsRequest.decoded(
            wireValue: try RestContractTestFixture.wireValue("""
            {
                "model": "mlx-community/nomicai-modernbert-embed-base-8bit",
                "input": "O Romeo, Romeo, wherefore art thou Romeo?"
            }
            """));
        let requestParts: OpenAiEmbeddingsRequestParts = try request.intoParts();
        XCTAssertEqual(requestParts.inputs, ["O Romeo, Romeo, wherefore art thou Romeo?"]);
        XCTAssertEqual(requestParts.encodingFormat, .float);
        XCTAssertNil(requestParts.dimensions);
    }

    func testShouldPreserveListOrderForAnArrayOfInputs() throws {
        let request: OpenAiEmbeddingsRequest = try OpenAiEmbeddingsRequest.decoded(
            wireValue: try RestContractTestFixture.wireValue("""
            {
                "model": "mlx-community/nomicai-modernbert-embed-base-8bit",
                "input": ["Two households, both alike in dignity.", "O Romeo, Romeo, wherefore art thou Romeo?"],
                "encoding_format": "base64",
                "dimensions": 256
            }
            """));
        let requestParts: OpenAiEmbeddingsRequestParts = try request.intoParts();
        XCTAssertEqual(requestParts.inputs.count, 2);
        XCTAssertEqual(requestParts.inputs[1], "O Romeo, Romeo, wherefore art thou Romeo?");
        XCTAssertEqual(requestParts.encodingFormat, .base64);
        XCTAssertEqual(requestParts.dimensions, 256);
    }

    func testShouldRejectAnEmptyInputList() throws {
        let request: OpenAiEmbeddingsRequest = try OpenAiEmbeddingsRequest.decoded(
            wireValue: try RestContractTestFixture.wireValue("""
            {
                "model": "mlx-community/nomicai-modernbert-embed-base-8bit",
                "input": []
            }
            """));
        do {
            _ = try request.intoParts();
            XCTFail("an empty input list must fail before queue admission");
            return;
        } catch let validationError as OpenAiEmbeddingsValidationError {
            XCTAssertTrue(validationError.errorDescription?.contains("at least one") == true);
        }
    }

    func testShouldRejectZeroDimensionsBeforeQueueAdmission() throws {
        let request: OpenAiEmbeddingsRequest = try OpenAiEmbeddingsRequest.decoded(
            wireValue: try RestContractTestFixture.wireValue("""
            {
                "model": "mlx-community/nomicai-modernbert-embed-base-8bit",
                "input": "O Romeo, Romeo, wherefore art thou Romeo?",
                "dimensions": 0
            }
            """));
        do {
            _ = try request.intoParts();
            XCTFail("dimensions 0 must fail before queue admission");
            return;
        } catch let validationError as OpenAiEmbeddingsValidationError {
            XCTAssertTrue(validationError.errorDescription?.contains("positive") == true);
        }
    }

    func testShouldRejectAnUnsupportedEncodingFormat() throws {
        let request: OpenAiEmbeddingsRequest = try OpenAiEmbeddingsRequest.decoded(
            wireValue: try RestContractTestFixture.wireValue("""
            {
                "model": "mlx-community/nomicai-modernbert-embed-base-8bit",
                "input": "hello",
                "encoding_format": "hex"
            }
            """));
        do {
            _ = try request.intoParts();
            XCTFail("unsupported encoding_format must fail closed");
            return;
        } catch let validationError as OpenAiEmbeddingsValidationError {
            XCTAssertTrue(validationError.errorDescription?.contains("hex") == true);
        }
    }

    func testShouldRejectAnOversizedEmbeddingInput() throws {
        let oversizedText: String = String(repeating: "x", count: 8_193);
        let request: OpenAiEmbeddingsRequest = try OpenAiEmbeddingsRequest.decoded(
            wireValue: try RestContractTestFixture.wireValue(
                #"{"model":"mlx-community/nomicai-modernbert-embed-base-8bit","input":"\#(oversizedText)"}"#));
        do {
            _ = try request.intoParts();
            XCTFail("an oversized single input must fail closed");
            return;
        } catch let validationError as OpenAiEmbeddingsValidationError {
            XCTAssertTrue(validationError.errorDescription?.contains("8193") == true);
        }
    }

    func testShouldRejectUnknownFieldsLikeMaxLengthForNow() throws {
        let request: OpenAiEmbeddingsRequest = try OpenAiEmbeddingsRequest.decoded(
            wireValue: try RestContractTestFixture.wireValue("""
            {
                "model": "mlx-community/nomicai-modernbert-embed-base-8bit",
                "input": "hello",
                "max_length": 512
            }
            """));
        do {
            _ = try request.intoParts();
            XCTFail("unknown fields must fail closed on the embeddings boundary");
            return;
        } catch let validationError as OpenAiEmbeddingsValidationError {
            XCTAssertTrue(validationError.errorDescription?.contains("max_length") == true);
        }
    }

    func testShouldSerializeAnOpenaiEmbeddingsListWithFloatRows() throws {
        let tokenUsage: OpenAiTokenUsage = try XCTUnwrap(
            OpenAiTokenUsage.new(promptTokens: 14, completionTokens: 0));
        let response: OpenAiEmbeddingsResponse = OpenAiEmbeddingsResponse(
            embeddings: [
                OpenAiEmbedding(index: 0, embedding: .float([1.0, 0.0])),
                OpenAiEmbedding(index: 1, embedding: .float([0.0, 1.0])),
            ],
            model: "mlx-community/nomicai-modernbert-embed-base-8bit",
            usage: tokenUsage);
        let serialized: String = try response.wireValue().serializedText;
        XCTAssertTrue(serialized.contains("\"object\":\"list\""),
            "embedding list must declare object list: \(serialized)");
        XCTAssertTrue(serialized.contains("\"object\":\"embedding\""),
            "each row must declare object embedding: \(serialized)");
        XCTAssertTrue(serialized.contains("\"index\":1"));
        XCTAssertTrue(serialized.contains("\"total_tokens\":14"));
    }

    func testShouldAdvertiseAnEmbeddingModelWithOnlyTheEmbeddingsEndpoint() throws {
        let model: OpenAiModel = try OpenAiModel.fromEmbeddingParts(
            embeddingModelParts: OpenAiEmbeddingModelParts(
                modelId: "nomicai-modernbert-embed-base-8bit",
                created: 1_784_231_803,
                ownedBy: "astronomical",
                vectorWidth: 768,
                maxInputTokens: 8_192));
        let serialized: String = try model.wireValue().serializedText;
        XCTAssertTrue(serialized.contains("\"supported_endpoints\":[\"/v1/embeddings\"]"));
        XCTAssertTrue(serialized.contains("\"output_modalities\":[\"embedding\"]"));
        XCTAssertTrue(serialized.contains("\"supports_streaming\":false"));
    }

    func testShouldRejectAZeroVectorWidthEmbeddingModel() throws {
        do {
            _ = try OpenAiModel.fromEmbeddingParts(
                embeddingModelParts: OpenAiEmbeddingModelParts(
                    modelId: "nomicai-modernbert-embed-base-8bit",
                    created: 1_784_231_803,
                    ownedBy: "astronomical",
                    vectorWidth: 0,
                    maxInputTokens: 8_192));
            XCTFail("a zero vector width must fail advertisement");
            return;
        } catch let validationError as OpenAiModelValidationError {
            XCTAssertTrue(validationError.errorDescription?.contains("vector width") == true);
        }
    }
}
