import Foundation;

import IpcProtocol;
import RestContract;

/**
 * One failure the REST transport or an endpoint answers with: an HTTP status
 * plus the shared OpenAI-compatible error envelope from RestContract.
 * Transport grammar failures (bad request line, oversized body) and endpoint
 * failures compose the same vocabulary — there are no endpoint-local error
 * shapes on the REST surface.
 */
public struct RestEndpointFailure: Error {

    public let statusCode: Int;
    public let errorResponse: OpenAiErrorResponse;

    public init(statusCode: Int, message: String) {
        self.statusCode = statusCode;
        self.errorResponse = OpenAiErrorResponse.invalidRequest(message: message, parameter: nil, code: nil);
    }

    private init(statusCode: Int, errorResponse: OpenAiErrorResponse) {
        self.statusCode = statusCode;
        self.errorResponse = errorResponse;
    }

    /// A handler or transport failure that is the server's fault, not the
    /// client's; answered with 500 and the server_error vocabulary.
    public static func internalError(message: String) -> RestEndpointFailure {
        return RestEndpointFailure(
            statusCode: 500,
            errorResponse: OpenAiErrorResponse.serviceUnavailable(message: message, code: nil));
    }

    /// Builds the HTTP response carrying this failure's envelope.
    public func envelopeResponse() throws -> RestHttpResponse {
        return try RestHttpResponse.json(statusCode: self.statusCode, wireValue: self.errorResponse.wireValue());
    }
}
