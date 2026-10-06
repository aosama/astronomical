import Foundation;
import RestContract;
import IpcProtocol;
import Testing;
import JourneyCategories;

/// Ported from crates/rest-contract/tests/rest_api/openai_embeddings.rs.
@Suite(.tags(.hermeticJourney))
final class EmbeddingsTests {

    @Test
    func should_validate_a_single_string_input_into_one_part() throws {
        let request: OpenAiEmbeddingsRequest = try OpenAiEmbeddingsRequest.decoded(
            wireValue: try RestContractTestFixture.wireValue("""
            {
                "model": "mlx-community/nomicai-modernbert-embed-base-8bit",
                "input": "O Romeo, Romeo, wherefore art thou Romeo?"
            }
            """));
        let requestParts: OpenAiEmbeddingsRequestParts = try request.intoParts();
        #expect(requestParts.inputs == ["O Romeo, Romeo, wherefore art thou Romeo?"]);
        #expect(requestParts.encodingFormat == .float);
        #expect(requestParts.dimensions == nil);
    }

    @Test
    func should_preserve_list_order_for_an_array_of_inputs() throws {
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
        #expect(requestParts.inputs.count == 2);
        #expect(requestParts.inputs[1] == "O Romeo, Romeo, wherefore art thou Romeo?");
        #expect(requestParts.encodingFormat == .base64);
        #expect(requestParts.dimensions == 256);
    }

    @Test
    func should_reject_an_empty_input_list() throws {
        let request: OpenAiEmbeddingsRequest = try OpenAiEmbeddingsRequest.decoded(
            wireValue: try RestContractTestFixture.wireValue("""
            {
                "model": "mlx-community/nomicai-modernbert-embed-base-8bit",
                "input": []
            }
            """));
        do {
            _ = try request.intoParts();
            Issue.record("an empty input list must fail before queue admission");
        } catch let validationError as OpenAiEmbeddingsValidationError {
            #expect(validationError.errorDescription?.contains("at least one") == true);
        }
    }

    @Test
    func should_reject_zero_dimensions_before_queue_admission() throws {
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
            Issue.record("dimensions 0 must fail before queue admission");
        } catch let validationError as OpenAiEmbeddingsValidationError {
            #expect(validationError.errorDescription?.contains("positive") == true);
        }
    }

    @Test
    func should_reject_an_unsupported_encoding_format() throws {
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
            Issue.record("unsupported encoding_format must fail closed");
        } catch let validationError as OpenAiEmbeddingsValidationError {
            #expect(validationError.errorDescription?.contains("hex") == true);
        }
    }

    @Test
    func should_reject_an_oversized_embedding_input() throws {
        let oversizedText: String = String(repeating: "x", count: 8_193);
        let request: OpenAiEmbeddingsRequest = try OpenAiEmbeddingsRequest.decoded(
            wireValue: try RestContractTestFixture.wireValue(
                #"{"model":"mlx-community/nomicai-modernbert-embed-base-8bit","input":"\#(oversizedText)"}"#));
        do {
            _ = try request.intoParts();
            Issue.record("an oversized single input must fail closed");
        } catch let validationError as OpenAiEmbeddingsValidationError {
            #expect(validationError.errorDescription?.contains("8193") == true);
        }
    }

    @Test
    func should_reject_unknown_fields_like_max_length_for_now() throws {
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
            Issue.record("unknown fields must fail closed on the embeddings boundary");
        } catch let validationError as OpenAiEmbeddingsValidationError {
            #expect(validationError.errorDescription?.contains("max_length") == true);
        }
    }

    @Test
    func should_serialize_an_openai_embeddings_list_with_float_rows() throws {
        let tokenUsage: OpenAiTokenUsage = try #require(
            OpenAiTokenUsage.new(promptTokens: 14, completionTokens: 0));
        let response: OpenAiEmbeddingsResponse = OpenAiEmbeddingsResponse(
            embeddings: [
                OpenAiEmbedding(index: 0, embedding: .float([1.0, 0.0])),
                OpenAiEmbedding(index: 1, embedding: .float([0.0, 1.0])),
            ],
            model: "mlx-community/nomicai-modernbert-embed-base-8bit",
            usage: tokenUsage);
        let serialized: String = try response.wireValue().serializedText;
        #expect(serialized.contains("\"object\":\"list\""),
            "embedding list must declare object list: \(serialized)");
        #expect(serialized.contains("\"object\":\"embedding\""),
            "each row must declare object embedding: \(serialized)");
        #expect(serialized.contains("\"index\":1"));
        #expect(serialized.contains("\"total_tokens\":14"));
    }

    @Test
    func should_advertise_an_embedding_model_with_only_the_embeddings_endpoint() throws {
        let model: OpenAiModel = try OpenAiModel.fromEmbeddingParts(
            embeddingModelParts: OpenAiEmbeddingModelParts(
                modelId: "nomicai-modernbert-embed-base-8bit",
                created: 1_784_231_803,
                ownedBy: "astronomical",
                vectorWidth: 768,
                maxInputTokens: 8_192));
        let serialized: String = try model.wireValue().serializedText;
        #expect(serialized.contains("\"supported_endpoints\":[\"/v1/embeddings\"]"));
        #expect(serialized.contains("\"output_modalities\":[\"embedding\"]"));
        #expect(serialized.contains("\"supports_streaming\":false"));
    }

    @Test
    func should_reject_a_zero_vector_width_embedding_model() throws {
        do {
            _ = try OpenAiModel.fromEmbeddingParts(
                embeddingModelParts: OpenAiEmbeddingModelParts(
                    modelId: "nomicai-modernbert-embed-base-8bit",
                    created: 1_784_231_803,
                    ownedBy: "astronomical",
                    vectorWidth: 0,
                    maxInputTokens: 8_192));
            Issue.record("a zero vector width must fail advertisement");
        } catch let validationError as OpenAiModelValidationError {
            #expect(validationError.errorDescription?.contains("vector width") == true);
        }
    }
}
