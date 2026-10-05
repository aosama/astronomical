import Foundation;
import Darwin;

/**
 * Weight measurement helpers for QwenImage21 discovery, ported from the
 * weight-accounting half of crates/config/src/model_discovery/qwen_image_21.rs.
 * Model size evidence must come from the reviewed component safetensors
 * indices and their shard files only.
 */
internal enum QwenImage21Weights {
    internal static func measureReviewedWeightBytes(modelDirectory: FilePath) throws -> UInt64 {
        var reviewedWeightSizeBytes: UInt64 = 0;
        for reviewedComponent: (componentDirectory: String, expectedClassName: String, componentErrorName: String) in QwenImage21.REVIEWED_COMPONENTS {
            let componentWeightSizeBytes: UInt64 = try self.measureComponentWeightBytes(
                modelDirectory: modelDirectory,
                componentDirectory: reviewedComponent.componentDirectory,
                componentErrorName: reviewedComponent.componentErrorName
            );
            let componentOverflowResult: (partialValue: UInt64, overflow: Bool) = reviewedWeightSizeBytes
                .addingReportingOverflow(componentWeightSizeBytes);
            if (componentOverflowResult.overflow) {
                throw QwenImage21.DirectoryVerificationError.modelSizeOverflow;
            }
            reviewedWeightSizeBytes = componentOverflowResult.partialValue;
        }
        return reviewedWeightSizeBytes;
    }

    private static func measureComponentWeightBytes(
        modelDirectory: FilePath,
        componentDirectory: String,
        componentErrorName: String
    ) throws -> UInt64 {
        let componentIndex: ComponentSafetensorsIndex;
        do {
            let indexObject: Dictionary<String, Any> = try QwenImage21.readJsonObject(
                documentPath: modelDirectory
                    .appending(component: componentDirectory)
                    .appending(component: "model.safetensors.index.json"),
                maximumBytes: QwenImage21.MAXIMUM_COMPONENT_INDEX_BYTES
            );
            componentIndex = try ComponentSafetensorsIndex.fromJsonObject(indexObject);
        } catch {
            throw QwenImage21.DirectoryVerificationError.invalidComponentWeightIndex(component: componentErrorName);
        }
        guard componentIndex.metadata.totalSize != 0 else {
            throw QwenImage21.DirectoryVerificationError.invalidComponentWeightIndex(component: componentErrorName);
        }
        var indexedShardPaths: Set<String> = Set<String>();
        for shardPath: String in componentIndex.weightMap.values {
            guard self.isSafeSafetensorsPath(shardPath: shardPath) else {
                throw QwenImage21.DirectoryVerificationError.invalidComponentWeightIndex(component: componentErrorName);
            }
            indexedShardPaths.insert(shardPath);
        }
        guard !indexedShardPaths.isEmpty else {
            throw QwenImage21.DirectoryVerificationError.invalidComponentWeightIndex(component: componentErrorName);
        }
        var componentWeightSizeBytes: UInt64 = 0;
        var componentPayloadSizeBytes: UInt64 = 0;
        // Sorted iteration mirrors the Rust BTreeSet's deterministic shard order.
        for shardPath: String in indexedShardPaths.sorted() {
            let indexedWeightPath: FilePath = modelDirectory
                .appending(component: componentDirectory)
                .appending(component: shardPath);
            let shardWeightSizeBytes: UInt64 = try self.requiredWeightSize(
                weightPath: indexedWeightPath,
                componentName: componentErrorName
            );
            let weightOverflowResult: (partialValue: UInt64, overflow: Bool) = componentWeightSizeBytes
                .addingReportingOverflow(shardWeightSizeBytes);
            if (weightOverflowResult.overflow) {
                throw QwenImage21.DirectoryVerificationError.modelSizeOverflow;
            }
            componentWeightSizeBytes = weightOverflowResult.partialValue;
            let shardPayloadSizeBytes: UInt64 = try self.requiredSafetensorsPayloadSize(
                weightPath: indexedWeightPath,
                componentName: componentErrorName
            );
            let payloadOverflowResult: (partialValue: UInt64, overflow: Bool) = componentPayloadSizeBytes
                .addingReportingOverflow(shardPayloadSizeBytes);
            if (payloadOverflowResult.overflow) {
                throw QwenImage21.DirectoryVerificationError.modelSizeOverflow;
            }
            componentPayloadSizeBytes = payloadOverflowResult.partialValue;
        }
        guard componentIndex.metadata.totalSize == componentPayloadSizeBytes else {
            throw QwenImage21.DirectoryVerificationError.invalidComponentWeightIndex(component: componentErrorName);
        }
        return componentWeightSizeBytes;
    }

    private static func requiredSafetensorsPayloadSize(weightPath: FilePath, componentName: String) throws -> UInt64 {
        var weightStatus: stat = stat();
        guard Darwin.fstatat(Darwin.AT_FDCWD, weightPath.string, &weightStatus, 0) == 0 else {
            throw QwenImage21.DirectoryVerificationError.invalidComponentWeightIndex(component: componentName);
        }
        let weightFileSizeBytes: UInt64 = UInt64(weightStatus.st_size);
        let weightFileHandle: FileHandle;
        do {
            weightFileHandle = try FileHandle(forReadingFrom: URL(fileURLWithPath: weightPath.string));
        } catch {
            throw QwenImage21.DirectoryVerificationError.invalidComponentWeightIndex(component: componentName);
        }
        defer {
            weightFileHandle.closeFile();
        }
        let headerLengthBytes: Data = weightFileHandle.readData(ofLength: 8);
        guard headerLengthBytes.count == 8 else {
            throw QwenImage21.DirectoryVerificationError.invalidComponentWeightIndex(component: componentName);
        }
        let headerSizeBytes: UInt64 = self.littleEndianUInt64(of: headerLengthBytes);
        let headerRemovalResult: (partialValue: UInt64, overflow: Bool) = weightFileSizeBytes.subtractingReportingOverflow(8);
        if (headerRemovalResult.overflow) {
            throw QwenImage21.DirectoryVerificationError.invalidComponentWeightIndex(component: componentName);
        }
        let payloadRemovalResult: (partialValue: UInt64, overflow: Bool) = headerRemovalResult.partialValue
            .subtractingReportingOverflow(headerSizeBytes);
        if (payloadRemovalResult.overflow) {
            throw QwenImage21.DirectoryVerificationError.invalidComponentWeightIndex(component: componentName);
        }
        guard payloadRemovalResult.partialValue > 0 else {
            throw QwenImage21.DirectoryVerificationError.invalidComponentWeightIndex(component: componentName);
        }
        return payloadRemovalResult.partialValue;
    }

    private static func requiredWeightSize(weightPath: FilePath, componentName: String) throws -> UInt64 {
        var weightStatus: stat = stat();
        guard Darwin.fstatat(Darwin.AT_FDCWD, weightPath.string, &weightStatus, 0) == 0 else {
            throw QwenImage21.DirectoryVerificationError.missingOrInvalidWeightFile(component: componentName);
        }
        guard (weightStatus.st_mode & S_IFMT) == S_IFREG else {
            throw QwenImage21.DirectoryVerificationError.missingOrInvalidWeightFile(component: componentName);
        }
        let weightSizeBytes: UInt64 = UInt64(weightStatus.st_size);
        guard weightSizeBytes > 0 else {
            throw QwenImage21.DirectoryVerificationError.missingOrInvalidWeightFile(component: componentName);
        }
        return weightSizeBytes;
    }

    private static func isSafeSafetensorsPath(shardPath: String) -> Bool {
        if shardPath.isEmpty || shardPath.contains("\\") {
            return false;
        }
        let shardFilePath: FilePath = FilePath(string: shardPath);
        if (shardFilePath.isAbsolute) {
            return false;
        }
        // Rust keeps only Normal components: a leading "." is CurDir and any
        // ".." is ParentDir, so both reject; interior "." normalizes away.
        let shardPathComponents: Array<String> = shardPath.split(
            omittingEmptySubsequences: true,
            whereSeparator: { (pathCharacter: Character) -> Bool in return pathCharacter == "/"; }
        ).map({ (pathComponent: Substring) -> String in return String(pathComponent); });
        for (offset: componentIndex, element: pathComponent) in shardPathComponents.enumerated() {
            if (componentIndex == 0 && pathComponent == ".") {
                return false;
            }
            if (pathComponent == "..") {
                return false;
            }
        }
        guard let shardFileName: String = DiscoveryPathNavigation.lastComponentName(of: shardFilePath) else {
            return false;
        }
        guard let extensionDotIndex: String.Index = shardFileName.lastIndex(of: ".") else {
            return false;
        }
        guard extensionDotIndex > shardFileName.startIndex else {
            return false;
        }
        let shardPathExtension: String = String(shardFileName[shardFileName.index(after: extensionDotIndex)...]);
        return shardPathExtension == "safetensors";
    }

    private static func littleEndianUInt64(of headerBytes: Data) -> UInt64 {
        var decodedSize: UInt64 = 0;
        for (offset: byteIndex, element: headerByte) in headerBytes.enumerated() {
            decodedSize = decodedSize | (UInt64(headerByte) << (8 * UInt64(byteIndex)));
        }
        return decodedSize;
    }
}
