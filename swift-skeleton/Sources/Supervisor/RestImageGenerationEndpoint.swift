import Foundation;

import AstronomicalConfig;
import IpcProtocol;
import RestContract;

/// Everything the REST image-generation route needs: the image executor
/// seam, the supervisor-local request-id space, and the live resolved
/// configuration the discovery and capability checks read from.
public struct RestImageGenerationRouteContext: @unchecked Sendable {

    let imageExecutor: any ImageGenerationExecuting;
    let requestIdAllocator: ChatRequestIdAllocator;
    let resolvedRuntimeConfig: ResolvedRuntimeConfig;

    public init(
        imageExecutor: any ImageGenerationExecuting,
        requestIdAllocator: ChatRequestIdAllocator,
        resolvedRuntimeConfig: ResolvedRuntimeConfig
    ) {
        self.imageExecutor = imageExecutor;
        self.requestIdAllocator = requestIdAllocator;
        self.resolvedRuntimeConfig = resolvedRuntimeConfig;
    }
}

/// The POST /v1/images/generations endpoint.
///
/// Port of apps/supervisor/src/openai_image_generation_endpoint.rs: complete
/// validation before shared admission — decode, public validation, health,
/// model resolution, capability and dimension-envelope checks — then one
/// image command whose diffusion schedule comes from discovery, and the
/// base64 answer with the metadata the worker reported.
enum RestImageGenerationEndpoint {

    static let routeMethod: String = "POST";
    static let routePath: String = "/v1/images/generations";

    static func handle(
        _ request: RestHttpRequest,
        imageContext: RestImageGenerationRouteContext
    ) -> RestHttpResponse {
        let imageRequest: OpenAiImageGenerationRequest;
        do {
            let requestWireValue: JsonWireValue = try JsonWireParser.parseDocument(
                documentBytes: request.bodyBytes);
            imageRequest = try OpenAiImageGenerationRequest.decoded(wireValue: requestWireValue);
        } catch {
            return RestImageGenerationEndpoint.invalidRequestResponse(
                message: "request body is not valid JSON: \(error)",
                parameter: nil,
                code: "invalid_json");
        }
        let requestParts: OpenAiImageGenerationRequestParts;
        do {
            requestParts = try imageRequest.intoParts();
        } catch let validationRejection as OpenAiImageGenerationValidationError {
            return RestImageGenerationEndpoint.invalidRequestResponse(
                message: validationRejection.errorDescription
                    ?? String(describing: validationRejection),
                parameter: RestImageGenerationEndpoint.validationParameter(validationRejection),
                code: "invalid_request");
        } catch {
            return RestImageGenerationEndpoint.invalidRequestResponse(
                message: String(describing: error),
                parameter: nil,
                code: "invalid_request");
        }
        let workerHealthSnapshot: WorkerHealthSnapshot =
            imageContext.imageExecutor.workerHealthSnapshot();
        if workerHealthSnapshot.status.isReady() == false {
            return RestImageGenerationEndpoint.workerUnavailableResponse();
        }
        let knownModelIds: Array<String> = imageContext.resolvedRuntimeConfig.discoveredModels
            .map { (discoveredModel: DiscoveryDiscoveredModel) -> String in
                return discoveredModel.modelId;
            };
        let resolvedModelId: String = ModelIdentity.resolveModelId(
            requestedModelId: requestParts.model,
            knownModelIds: knownModelIds);
        guard let discoveredModel: DiscoveryDiscoveredModel =
            imageContext.resolvedRuntimeConfig.discoveredModels.first(
                where: { (candidateModel: DiscoveryDiscoveredModel) -> Bool in
                    return candidateModel.modelId == resolvedModelId;
                }) else {
            return RestImageGenerationEndpoint.invalidRequestResponse(
                message: "model is not available to the local worker",
                parameter: "model",
                code: "model_not_found");
        }
        // Image workers start idle and publish capabilities only after the
        // request-triggered swap, so the diffusion schedule and the accepted
        // dimension grid are discovery facts, not worker state.
        guard case let .imageGeneration(imageCapabilities) = discoveredModel.capabilities,
              imageCapabilities.supportsTextToImage else {
            return RestImageGenerationEndpoint.invalidRequestResponse(
                message: "the requested model does not support text-to-image generation",
                parameter: "model",
                code: "model_capability_mismatch");
        }
        // A doomed request must never win the active-generation permit or
        // trigger a model swap the capability check would immediately undo.
        if let dimensionViolation: (parameterName: String, violationMessage: String) =
            imageCapabilities.imageDimensionViolation(
                modelId: resolvedModelId,
                widthPixels: requestParts.width,
                heightPixels: requestParts.height) {
            return RestImageGenerationEndpoint.invalidRequestResponse(
                message: dimensionViolation.violationMessage,
                parameter: dimensionViolation.parameterName,
                code: "invalid_request");
        }
        guard let requestIdentifier: UInt64 = imageContext.requestIdAllocator.allocate() else {
            return RestImageGenerationEndpoint.serviceUnavailableResponse(
                message: "the local request identifier space is exhausted",
                code: "request_id_exhausted");
        }
        let imageGenerationCommand: ImageGenerationCommand = ImageGenerationCommand(
            requestId: RequestId(rawRequestId: requestIdentifier),
            model: resolvedModelId,
            prompt: requestParts.prompt,
            settings: ImageGenerationSettings(
                widthPixels: requestParts.width,
                heightPixels: requestParts.height,
                steps: imageCapabilities.defaultSteps,
                guidanceThousandths: 1_000,
                seed: requestParts.seed
                    ?? RestImageGenerationEndpoint.generatedSeed(requestIdentifier)));
        let imageOutput: ImageGenerationOutput;
        do {
            imageOutput = try imageContext.imageExecutor.startImageGeneration(
                imageGenerationCommand);
        } catch let startError as GenerationStartError {
            return RestImageGenerationEndpoint.generationStartFailureResponse(startError);
        } catch let executionError as ImageGenerationExecutionError {
            return RestImageGenerationEndpoint.executionFailureResponse(executionError);
        } catch {
            return RestImageGenerationEndpoint.workerUnavailableResponse();
        }
        guard let createdAtUnixSeconds: UInt64 = RestImageGenerationEndpoint.currentUnixSeconds() else {
            return RestImageGenerationEndpoint.workerUnavailableResponse();
        }
        let imageResponse: OpenAiImageGenerationResponse = OpenAiImageGenerationResponse(
            created: createdAtUnixSeconds,
            generatedImageParts: OpenAiGeneratedImageParts(
                b64Json: Data(imageOutput.generatedImage.encodedBytes).base64EncodedString(),
                mimeType: imageOutput.generatedImage.mimeType,
                modelRevision: discoveredModel.revision,
                effectiveSeed: imageOutput.resultMetadata.seed,
                width: imageOutput.resultMetadata.widthPixels,
                height: imageOutput.resultMetadata.heightPixels));
        return (try? RestHttpResponse.json(
            statusCode: 200,
            wireValue: imageResponse.wireValue()))
            ?? RestHttpResponse.text(statusCode: 500, body: "the response could not be serialized");
    }

    private static func validationParameter(
        _ validationError: OpenAiImageGenerationValidationError
    ) -> String {
        switch (validationError) {
        case .unknownField:
            return "request";
        case .blankModel:
            return "model";
        case .blankPrompt:
            return "prompt";
        case let .unsupportedDimension(parameterName, _, _, _):
            return parameterName;
        case .unsupportedResponseFormat:
            return "response_format";
        case .unsupportedImageCount:
            return "n";
        }
    }

    /// The deterministic-per-request seed: time mixed with the request
    /// position, mirroring the Rust generated_seed rotation.
    private static func generatedSeed(_ requestIdentifier: UInt64) -> UInt64 {
        let secondsSinceEpoch: TimeInterval = Date().timeIntervalSince1970;
        guard secondsSinceEpoch >= 0 else {
            return requestIdentifier << 32;
        }
        let wholeSeconds: UInt64 = UInt64(secondsSinceEpoch);
        let subsecondNanoseconds: UInt64 = UInt64(
            (secondsSinceEpoch - Double(wholeSeconds)) * 1_000_000_000);
        let timeSeed: UInt64 = ((wholeSeconds << 32) | (wholeSeconds >> 32)) ^ subsecondNanoseconds;
        return timeSeed ^ (requestIdentifier << 17 | requestIdentifier >> 47);
    }

    private static func currentUnixSeconds() -> UInt64? {
        let secondsSinceEpoch: TimeInterval = Date().timeIntervalSince1970;
        guard secondsSinceEpoch >= 0 else {
            return nil;
        }
        return UInt64(secondsSinceEpoch);
    }

    private static func generationStartFailureResponse(
        _ startError: GenerationStartError
    ) -> RestHttpResponse {
        switch (startError) {
        case .capacityUnavailable:
            return RestImageGenerationEndpoint.capacityResponse(
                message: "the generation queue is full");
        case let .modelLoadFailed(modelLoadFailureReason):
            return RestImageGenerationEndpoint.jsonResponse(
                statusCode: 503,
                errorResponse: OpenAiErrorResponse.modelLoadFailed(
                    modelLoadFailureReason: modelLoadFailureReason));
        case .requestTooLarge:
            return RestImageGenerationEndpoint.invalidRequestResponse(
                message: "the image request exceeds the local worker transport limit",
                parameter: nil,
                code: "request_too_large",
                statusCodeOverride: 413);
        case .workerUnavailable:
            return RestImageGenerationEndpoint.workerUnavailableResponse();
        }
    }

    private static func executionFailureResponse(
        _ executionError: ImageGenerationExecutionError
    ) -> RestHttpResponse {
        switch (executionError) {
        case let .workerFailure(failureReason):
            switch (failureReason) {
            case .invalidRequest:
                return RestImageGenerationEndpoint.invalidRequestResponse(
                    message: "the image request was rejected by the local worker",
                    parameter: nil,
                    code: "invalid_request");
            case .modelDoesNotSupportImageGeneration:
                return RestImageGenerationEndpoint.invalidRequestResponse(
                    message: "the requested model does not support image generation",
                    parameter: "model",
                    code: "model_capability_mismatch");
            case .engineBusy:
                return RestImageGenerationEndpoint.capacityResponse(
                    message: "the image engine is busy");
            case .encodingFailed, .fatalExecution:
                return RestImageGenerationEndpoint.imageWorkerFailureResponse();
            case .cancelled:
                return RestImageGenerationEndpoint.workerUnavailableResponse();
            }
        case .workerUnavailable:
            return RestImageGenerationEndpoint.workerUnavailableResponse();
        case .deadlineExceeded:
            return RestImageGenerationEndpoint.jsonResponse(
                statusCode: 504,
                errorResponse: OpenAiErrorResponse.serviceUnavailable(
                    message: "image generation exceeded its bounded execution deadline",
                    code: "image_generation_timeout"));
        }
    }

    private static func imageWorkerFailureResponse() -> RestHttpResponse {
        return RestImageGenerationEndpoint.jsonResponse(
            statusCode: 500,
            errorResponse: OpenAiErrorResponse.serviceUnavailable(
                message: "image generation failed in the local worker",
                code: "image_generation_failed"));
    }

    private static func capacityResponse(message: String) -> RestHttpResponse {
        return RestImageGenerationEndpoint.jsonResponse(
            statusCode: 429,
            errorResponse: OpenAiErrorResponse.capacityUnavailable(message: message));
    }

    private static func workerUnavailableResponse() -> RestHttpResponse {
        return RestImageGenerationEndpoint.serviceUnavailableResponse(
            message: "the local worker is unavailable",
            code: "worker_unavailable");
    }

    private static func serviceUnavailableResponse(
        message: String,
        code: String
    ) -> RestHttpResponse {
        return RestImageGenerationEndpoint.jsonResponse(
            statusCode: 503,
            errorResponse: OpenAiErrorResponse.serviceUnavailable(message: message, code: code));
    }

    private static func invalidRequestResponse(
        message: String,
        parameter: String?,
        code: String,
        statusCodeOverride: Int? = nil
    ) -> RestHttpResponse {
        return RestImageGenerationEndpoint.jsonResponse(
            statusCode: statusCodeOverride ?? 400,
            errorResponse: OpenAiErrorResponse.invalidRequest(
                message: message, parameter: parameter, code: code));
    }

    private static func jsonResponse(
        statusCode: Int,
        errorResponse: OpenAiErrorResponse
    ) -> RestHttpResponse {
        return (try? RestHttpResponse.json(
            statusCode: statusCode,
            wireValue: errorResponse.wireValue()))
            ?? RestHttpResponse.text(statusCode: 500, body: "the response could not be serialized");
    }
}
