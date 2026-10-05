import Foundation;

/// Data-URI image input decoding for the OpenAI-compatible chat endpoint.
///
/// Only `data:image/<type>;base64,<payload>` URIs are accepted. HTTP(S) and
/// `file://` schemes are rejected to preserve the single-laptop privacy model
/// and avoid local-path attack surface. Port of crates/rest-contract/src/image_input.rs.
public enum ImageInput {

    /// Maximum accepted decoded image byte payload. Matches the chat body cap.
    public static let MAX_OPENAI_IMAGE_BYTES: Int = 32 * 1024 * 1024;

    /// One decoded image carried in a user chat message.
    public struct OpenAiImageInput: Equatable {
        /// The MIME type parsed from the data URI, e.g. `image/png`.
        private let imageMimeType: String;
        /// The raw decoded image file bytes (PNG/JPEG/WebP payload before pixel decoding).
        private let decodedImageBytes: Array<UInt8>;

        fileprivate init(imageMimeType: String, decodedImageBytes: Array<UInt8>) {
            self.imageMimeType = imageMimeType;
            self.decodedImageBytes = decodedImageBytes;
        }

        /// Returns the MIME type parsed from the data URI.
        public func mimeType() -> String {
            return self.imageMimeType;
        }

        /// Returns the raw decoded image file bytes.
        public func decodedBytes() -> Array<UInt8> {
            return self.decodedImageBytes;
        }
    }

    /// Decodes a pre-validated `data:image/<type>;base64,<payload>` URI into image bytes.
    ///
    /// The caller must have already validated the scheme and MIME type via
    /// `validateImageUrlScheme`. This function extracts the MIME type, decodes
    /// the base64 payload, and enforces the byte size bound.
    public static func decodeImageUrl(imageUrl: String) throws -> OpenAiImageInput {
        guard imageUrl.hasPrefix("data:") else {
            throw OpenAiChatCompletionValidationError.unsupportedImageUrlScheme;
        }
        let dataUriBody: String = String(imageUrl.dropFirst("data:".count));
        guard let commaPosition: String.Index = dataUriBody.firstIndex(of: ",") else {
            throw OpenAiChatCompletionValidationError.malformedDataUri;
        }
        let metadata: String = String(dataUriBody[dataUriBody.startIndex..<commaPosition]);
        let base64Payload: String = String(dataUriBody[dataUriBody.index(after: commaPosition)...]);
        guard let separatorPosition: String.Index = metadata.firstIndex(of: ";") else {
            throw OpenAiChatCompletionValidationError.malformedDataUri;
        }
        let imageMimeType: String = String(metadata[metadata.startIndex..<separatorPosition]);
        let decodedImageBytes: Array<UInt8> = try decodeStrictBase64(base64Payload: base64Payload);
        if decodedImageBytes.count > MAX_OPENAI_IMAGE_BYTES {
            throw OpenAiChatCompletionValidationError.imageTooLarge(
                actualBytes: decodedImageBytes.count, maximumBytes: MAX_OPENAI_IMAGE_BYTES);
        }
        return OpenAiImageInput(imageMimeType: imageMimeType, decodedImageBytes: decodedImageBytes);
    }

    /// Validates a data-URI image before decoding its bounded payload.
    public static func validateImageUrlScheme(imageUrl: String) throws -> Void {
        guard imageUrl.hasPrefix("data:") else {
            throw OpenAiChatCompletionValidationError.unsupportedImageUrlScheme;
        }
        let dataUriBody: String = String(imageUrl.dropFirst("data:".count));
        guard let commaPosition: String.Index = dataUriBody.firstIndex(of: ",") else {
            throw OpenAiChatCompletionValidationError.malformedDataUri;
        }
        let metadata: String = String(dataUriBody[dataUriBody.startIndex..<commaPosition]);
        let base64Payload: String = String(dataUriBody[dataUriBody.index(after: commaPosition)...]);
        guard let separatorPosition: String.Index = metadata.firstIndex(of: ";") else {
            throw OpenAiChatCompletionValidationError.malformedDataUri;
        }
        let mimeType: String = String(metadata[metadata.startIndex..<separatorPosition]);
        let encoding: String = String(metadata[metadata.index(after: separatorPosition)...]);
        if encoding != "base64" {
            throw OpenAiChatCompletionValidationError.unsupportedImageUrlScheme;
        }
        if mimeType.hasPrefix("image/") == false {
            throw OpenAiChatCompletionValidationError.unsupportedImageMimeType(actualMimeType: mimeType);
        }
        for payloadScalar in base64Payload.unicodeScalars {
            if isAllowedBase64Scalar(payloadScalar) == false {
                throw OpenAiChatCompletionValidationError.invalidBase64;
            }
        }
    }

    /// The Rust port uses the base64 STANDARD engine, which accepts only the
    /// canonical alphabet plus padding; Foundation's decoder is more lenient
    /// about missing padding, so the charset gate here plus the padded-length
    /// check in `decodeStrictBase64` preserve the exact rejection boundary.
    private static func isAllowedBase64Scalar(_ scalar: Unicode.Scalar) -> Bool {
        if (scalar.value >= 65 && scalar.value <= 90)
            || (scalar.value >= 97 && scalar.value <= 122)
            || (scalar.value >= 48 && scalar.value <= 57) {
            return true;
        }
        return scalar == "+" || scalar == "/" || scalar == "=";
    }

    private static func decodeStrictBase64(base64Payload: String) throws -> Array<UInt8> {
        if base64Payload.count % 4 != 0 {
            throw OpenAiChatCompletionValidationError.invalidBase64;
        }
        guard let decodedData: Data = Data(base64Encoded: base64Payload) else {
            throw OpenAiChatCompletionValidationError.invalidBase64;
        }
        return Array<UInt8>(decodedData);
    }
}
