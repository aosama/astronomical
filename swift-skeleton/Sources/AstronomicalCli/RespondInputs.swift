import Foundation

import IpcProtocol;

/**
 * Reads and validates the `--image` and `--schema` inputs for
 * `astronomical respond`, porting respond_image.rs and respond_schema.rs.
 * Local inputs fail before any daemon work: a missing schema file or an
 * unreadable image must never start a model download.
 */
public enum RespondInputs {

    /// Decoded image bytes the CLI accepts in total for one request. This
    /// leaves headroom for the base64 inflation and framing the IPC frame
    /// carries, so a passing check cannot exceed the frame limit.
    public static let maximumTotalImageDecodedBytes: Int = 16 * 1024 * 1024;

    /// Supported image extensions paired with the wire MIME type.
    private static let supportedImageFormats: Array<(extensionName: String, mimeType: String)> = [
        ("png", "image/png"),
        ("jpg", "image/jpeg"),
        ("jpeg", "image/jpeg"),
        ("webp", "image/webp"),
    ];

    /// The supported image extensions, for error messages.
    public static var supportedImageExtensions: String {
        return "png, jpg, jpeg, webp";
    }

    /// Reads and validates every image file for a request into decoded
    /// inputs, enforcing the shared decoded-byte budget across all images.
    public static func readImageInputs(
        _ paths: Array<String>
    ) throws -> Array<ChatImageInput> {
        var imageInputs: Array<ChatImageInput> = [];
        imageInputs.reserveCapacity(paths.count);
        var totalBytes: Int = 0;
        for imagePath: String in paths {
            let imageInput: ChatImageInput = try RespondInputs.readImageInput(path: imagePath);
            totalBytes += imageInput.decodedBytes.count;
            if (totalBytes > RespondInputs.maximumTotalImageDecodedBytes) {
                throw RespondError.imageTooLarge(
                    actualBytes: totalBytes,
                    maximumBytes: RespondInputs.maximumTotalImageDecodedBytes
                );
            }
            imageInputs.append(imageInput);
        }
        return imageInputs;
    }

    /// Reads one image file into a decoded IPC input, or explains why it is
    /// rejected.
    public static func readImageInput(path: String) throws -> ChatImageInput {
        let decodedBytes: Data;
        do {
            decodedBytes = try Data(contentsOf: URL(fileURLWithPath: path));
        } catch let readError {
            throw RespondError.imageReadFailed(path: path, cause: readError.localizedDescription);
        }
        guard let mimeType: String = RespondInputs.mimeForExtension(path: path) else {
            throw RespondError.unsupportedImage(
                path: path,
                supported: RespondInputs.supportedImageExtensions
            );
        }
        return ChatImageInput(mimeType: mimeType, decodedBytes: Array(decodedBytes));
    }

    /// Maps an image file extension to its wire MIME type, ignoring case.
    public static func mimeForExtension(path: String) -> String? {
        let fileExtension: String = ((path as NSString).pathExtension).lowercased();
        return RespondInputs.supportedImageFormats.first { (supportedFormat: (extensionName: String, mimeType: String)) -> Bool in
            return supportedFormat.extensionName == fileExtension;
        }?.mimeType;
    }

    /// Reads the schema file into the raw JSON text the daemon request
    /// carries. Schema validation itself is the daemon's job.
    public static func readSchemaInput(schemaPath: String) throws -> String {
        let schemaBytes: Data;
        do {
            schemaBytes = try Data(contentsOf: URL(fileURLWithPath: schemaPath));
        } catch let readError {
            throw RespondError.schemaReadFailed(path: schemaPath, cause: readError.localizedDescription);
        }
        if (schemaBytes.count > StructuredGenerationConstraint.maximumChatSchemaJsonBytes) {
            throw RespondError.schemaTooLarge(
                actualBytes: schemaBytes.count,
                maximumBytes: StructuredGenerationConstraint.maximumChatSchemaJsonBytes
            );
        }
        guard let schemaText: String = String(data: schemaBytes, encoding: .utf8) else {
            throw RespondError.schemaNotUtf8(
                path: schemaPath,
                cause: "the schema bytes are not valid UTF-8"
            );
        }
        return schemaText;
    }
}
