import Foundation;
import RestContract;
import IpcProtocol;
import Testing;
import JourneyCategories;

/// Ported from crates/rest-contract/tests/rest_api/openai_image_generation.rs.
@Suite(.tags(.hermeticJourney))
final class ImageGenerationTests {

    @Test
    func should_validate_a_complete_image_generation_request_into_effective_parts() throws {
        let request: OpenAiImageGenerationRequest = try Self.deserializeRequest("""
        {
            "model":"black-forest-labs/FLUX.2-klein-4B",
            "prompt":"A moonlit balcony scene from Romeo and Juliet.",
            "seed":18446744073709551615,
            "width":1024,
            "height":768,
            "response_format":"b64_json",
            "n":1
        }
        """);
        let requestParts: OpenAiImageGenerationRequestParts = try request.intoParts();
        #expect(requestParts.model == "black-forest-labs/FLUX.2-klein-4B");
        #expect(requestParts.prompt == "A moonlit balcony scene from Romeo and Juliet.");
        #expect(requestParts.seed == UInt64.max);
        #expect(requestParts.width == 1_024);
        #expect(requestParts.height == 768);
        #expect(requestParts.responseFormat == .base64Json);
        #expect(requestParts.imageCount == 1);
    }

    @Test
    func should_default_the_image_count_and_optional_seed() throws {
        let request: OpenAiImageGenerationRequest = try Self.deserializeRequest(
            Self.validRequestFields(width: "64", height: "64"));
        let requestParts: OpenAiImageGenerationRequestParts = try request.intoParts();
        #expect(requestParts.imageCount == 1);
        #expect(requestParts.seed == nil);
    }

    @Test
    func should_reject_step_and_guidance_fields_as_unknown() throws {
        // The diffusion schedule is owned by the worker's model profile, so a caller that still
        // sends the removed control fields gets the strict unknown-field rejection rather than a
        // silently ignored knob.
        for intrudingFieldName in ["steps", "guidance"] {
            let requestJson: String = "{\"model\":\"flux\",\"prompt\":\"A rose.\",\"width\":64,\"height\":64,\"response_format\":\"b64_json\",\"\(intrudingFieldName)\":1}";
            let request: OpenAiImageGenerationRequest = try Self.deserializeRequest(requestJson);
            do {
                _ = try request.intoParts();
                Issue.record("intruding field \(intrudingFieldName) must fail closed");
            } catch let validationError as OpenAiImageGenerationValidationError {
                #expect(
                    validationError
                        == OpenAiImageGenerationValidationError.unknownField(fieldName: intrudingFieldName));
            }
        }
    }

    @Test
    func should_reject_blank_or_non_string_prompts() throws {
        let blankPromptRequest: OpenAiImageGenerationRequest = try Self.deserializeRequest(
            #"{"model":"flux","prompt":" \n\t","width":64,"height":64,"response_format":"b64_json"}"#);
        #expect(
            Self.intoPartsResult(blankPromptRequest)
                == Result<OpenAiImageGenerationRequestParts, OpenAiImageGenerationValidationError>
                    .failure(.blankPrompt));
        Self.assertMalformedRequest(
            #"{"model":"flux","prompt":7,"width":64,"height":64,"response_format":"b64_json"}"#);
    }

    @Test
    func should_reject_dimensions_outside_the_supported_geometry() throws {
        let invalidDimensions: Array<(width: String, height: String, parameterName: String, actualPixels: UInt32)> = [
            ("0", "64", "width", 0),
            ("48", "64", "width", 48),
            ("65", "64", "width", 65),
            ("64", "1040", "height", 1_040),
        ];
        for invalidDimension in invalidDimensions {
            let request: OpenAiImageGenerationRequest = try Self.deserializeRequest(
                Self.validRequestFields(width: invalidDimension.width, height: invalidDimension.height));
            #expect(
                Self.intoPartsResult(request)
                    == Result<OpenAiImageGenerationRequestParts, OpenAiImageGenerationValidationError>
                        .failure(.unsupportedDimension(
                            parameterName: invalidDimension.parameterName,
                            actualPixels: invalidDimension.actualPixels,
                            minimumPixels: 64,
                            maximumPixels: 1_024)));
        }
    }

    @Test
    func should_reject_malformed_dimension_transport_values() {
        for malformedFields in [
            Self.validRequestFields(width: "64.5", height: "64"),
            Self.validRequestFields(width: "-64", height: "64"),
            Self.validRequestFields(width: "4294967296", height: "64"),
        ] {
            Self.assertMalformedRequest(malformedFields);
        }
    }

    @Test
    func should_reject_malformed_seed_transport_values() {
        for malformedSeed in ["-1", "1.5", "18446744073709551616", "\"7\""] {
            let requestJson: String = "{\"model\":\"flux\",\"prompt\":\"A rose.\",\"seed\":\(malformedSeed),\"width\":64,\"height\":64,\"response_format\":\"b64_json\"}";
            Self.assertMalformedRequest(requestJson);
        }
    }

    @Test
    func should_reject_unsupported_format_count_and_unknown_fields() throws {
        let invalidRequests: Array<(requestJson: String, expectedError: OpenAiImageGenerationValidationError)> = [
            (
                #"{"model":"flux","prompt":"A rose.","width":64,"height":64,"response_format":"url"}"#,
                .unsupportedResponseFormat(responseFormat: "url")
            ),
            (
                #"{"model":"flux","prompt":"A rose.","width":64,"height":64,"response_format":"b64_json","n":2}"#,
                .unsupportedImageCount(actualImages: 2)
            ),
            (
                #"{"model":"flux","prompt":"A rose.","width":64,"height":64,"response_format":"b64_json","quality":"hd"}"#,
                .unknownField(fieldName: "quality")
            ),
        ];
        for invalidRequest in invalidRequests {
            let request: OpenAiImageGenerationRequest = try Self.deserializeRequest(invalidRequest.requestJson);
            #expect(
                Self.intoPartsResult(request)
                    == Result<OpenAiImageGenerationRequestParts, OpenAiImageGenerationValidationError>
                        .failure(invalidRequest.expectedError));
        }
    }

    @Test
    func should_serialize_one_generated_image_with_reproducibility_metadata() throws {
        let response: OpenAiImageGenerationResponse = OpenAiImageGenerationResponse(
            created: 1_787_010_400,
            generatedImageParts: OpenAiGeneratedImageParts(
                b64Json: "iVBORw0KGgoAAAANSUhEUg==",
                mimeType: "image/png",
                modelRevision: "0123456789abcdef",
                effectiveSeed: UInt64.max,
                width: 1_024,
                height: 768));
        #expect(
            try response.wireValue().serializedText
                == #"{"created":1787010400,"data":[{"b64_json":"iVBORw0KGgoAAAANSUhEUg==","mime_type":"image/png","model_revision":"0123456789abcdef","seed":18446744073709551615,"width":1024,"height":768}]}"#);
    }

    private static func deserializeRequest(_ requestJson: String) throws -> OpenAiImageGenerationRequest {
        return try OpenAiImageGenerationRequest.decoded(
            wireValue: try RestContractTestFixture.wireValue(requestJson));
    }

    private static func assertMalformedRequest(_ requestJson: String) {
        do {
            _ = try OpenAiImageGenerationRequest.decoded(
                wireValue: try RestContractTestFixture.wireValue(requestJson));
            Issue.record("the malformed transport value must be rejected during deserialization");
        } catch {
            // Expected: malformed transport values fail at the decode boundary.
        }
    }

    private static func validRequestFields(width: String, height: String) -> String {
        return "{\"model\":\"flux\",\"prompt\":\"A rose.\",\"width\":\(width),\"height\":\(height),\"response_format\":\"b64_json\"}";
    }

    private static func intoPartsResult(
        _ request: OpenAiImageGenerationRequest) -> Result<OpenAiImageGenerationRequestParts, OpenAiImageGenerationValidationError> {
        do {
            return .success(try request.intoParts());
        } catch let validationError as OpenAiImageGenerationValidationError {
            return .failure(validationError);
        } catch {
            Issue.record("unexpected error type: \(error)");
            return .failure(.blankModel);
        }
    }
}
