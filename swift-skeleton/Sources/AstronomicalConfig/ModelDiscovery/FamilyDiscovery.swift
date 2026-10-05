import Foundation;
import Darwin;

/**
 * Family classification of a model directory from its marker document:
 * `model_index.json` when present, otherwise `config.json`. Size limits are
 * enforced from file metadata before any read so an oversized marker is never
 * buffered, mirroring the Rust dispatch order across family verifiers.
 */
internal enum FamilyDiscovery {
    private static let maximumConfigBytes: UInt64 = 4 * 1024 * 1024;
    private static let maximumPipelineIndexBytes: UInt64 = 1 * 1024 * 1024;

    internal static func classifyModelDirectory(
        modelDirectory: FilePath,
        attributionEnabled: Bool
    ) throws -> ModelFamily? {
        return try DiscoveryPerformanceTrace.measure(
            operationName: "FamilyDiscovery.classifyModelDirectory",
            attributionEnabled: attributionEnabled,
            operation: { () throws -> ModelFamily? in
                if DiscoveryPathNavigation.isExistingRegularFile(path: modelDirectory.appending(component: "model_index.json")) {
                    return try self.classifyPipelineDirectory(modelDirectory: modelDirectory);
                }
                return try self.classifyConfigDirectory(modelDirectory: modelDirectory);
            }
        );
    }

    private static func classifyConfigDirectory(modelDirectory: FilePath) throws -> ModelFamily? {
        let configPath: FilePath = modelDirectory.appending(component: "config.json");
        let configBytes: Data = try self.readMarkerDocument(
            documentPath: configPath,
            maximumBytes: self.maximumConfigBytes,
            readFailure: { (underlyingError: any Error) -> ModelFamilyClassificationError in
                ModelFamilyClassificationError.readConfig(modelDirectory: modelDirectory, underlyingError: underlyingError);
            },
            oversizeFailure: { (actualBytes: UInt64) -> ModelFamilyClassificationError in
                ModelFamilyClassificationError.configTooLarge(actualBytes: actualBytes, maximumBytes: self.maximumConfigBytes);
            }
        );
        let parsedConfig: Any;
        do {
            parsedConfig = try DiscoveryStrictJsonDocument.parseDocument(bytes: configBytes);
        } catch let parseError as DiscoveryStrictJsonDocument.ParseError {
            throw ModelFamilyClassificationError.parseConfig(
                modelDirectory: modelDirectory,
                underlyingDescription: parseError.description
            );
        }
        guard let configObject: Dictionary<String, Any> = parsedConfig as? Dictionary<String, Any> else {
            throw ModelFamilyClassificationError.parseConfig(
                modelDirectory: modelDirectory,
                underlyingDescription: "expected a JSON object"
            );
        }
        return ModelFamily.fromModelType(try StrictJson.optionalString(object: configObject, fieldName: "model_type"));
    }

    private static func classifyPipelineDirectory(modelDirectory: FilePath) throws -> ModelFamily? {
        let pipelineIndexPath: FilePath = modelDirectory.appending(component: "model_index.json");
        let pipelineIndexBytes: Data = try self.readMarkerDocument(
            documentPath: pipelineIndexPath,
            maximumBytes: self.maximumPipelineIndexBytes,
            readFailure: { (underlyingError: any Error) -> ModelFamilyClassificationError in
                ModelFamilyClassificationError.readPipelineIndex(modelDirectory: modelDirectory, underlyingError: underlyingError);
            },
            oversizeFailure: { (actualBytes: UInt64) -> ModelFamilyClassificationError in
                ModelFamilyClassificationError.pipelineIndexTooLarge(actualBytes: actualBytes, maximumBytes: self.maximumPipelineIndexBytes);
            }
        );
        do {
            if try Flux2Klein.classifiesPipelineIndex(pipelineIndexBytes) {
                return ModelFamily.flux2Klein;
            }
        } catch {
            throw ModelFamilyClassificationError.parsePipelineIndex(
                modelDirectory: modelDirectory,
                underlyingDescription: "\(error)"
            );
        }
        do {
            if try QwenImage21.classifiesPipelineIndex(pipelineIndexBytes) {
                return ModelFamily.qwenImage21;
            }
        } catch {
            throw ModelFamilyClassificationError.parsePipelineIndex(
                modelDirectory: modelDirectory,
                underlyingDescription: "\(error)"
            );
        }
        return nil;
    }

    /** Shared by other family dispatchers that read marker documents at their own limits. */
    internal static func classifyPipelineIndexBytes(pipelineIndexBytes: Data) throws -> ModelFamily? {
        if try Flux2Klein.classifiesPipelineIndex(pipelineIndexBytes) {
            return ModelFamily.flux2Klein;
        }
        if try QwenImage21.classifiesPipelineIndex(pipelineIndexBytes) {
            return ModelFamily.qwenImage21;
        }
        return nil;
    }

    private static func readMarkerDocument(
        documentPath: FilePath,
        maximumBytes: UInt64,
        readFailure: (any Error) -> ModelFamilyClassificationError,
        oversizeFailure: (UInt64) -> ModelFamilyClassificationError
    ) throws -> Data {
        var documentStatus: stat = stat();
        guard Darwin.fstatat(Darwin.AT_FDCWD, documentPath.string, &documentStatus, 0) == 0 else {
            throw readFailure(NSError(
                domain: NSPOSIXErrorDomain,
                code: Int(errno),
                userInfo: [NSLocalizedDescriptionKey: String(cString: strerror(errno))]
            ));
        }
        let documentSizeBytes: UInt64 = UInt64(documentStatus.st_size);
        guard documentSizeBytes <= maximumBytes else {
            throw oversizeFailure(documentSizeBytes);
        }
        let documentHandle: FileHandle;
        do {
            documentHandle = try FileHandle(forReadingFrom: URL(fileURLWithPath: documentPath.string));
        } catch {
            throw readFailure(error);
        }
        defer {
            documentHandle.closeFile();
        }
        let documentBytes: Data;
        do {
            documentBytes = try documentHandle.read(upToCount: Int(maximumBytes) + 1) ?? Data();
        } catch {
            throw readFailure(error);
        }
        if documentBytes.count > Int(maximumBytes) {
            throw oversizeFailure(UInt64(documentBytes.count));
        }
        return documentBytes;
    }
}
