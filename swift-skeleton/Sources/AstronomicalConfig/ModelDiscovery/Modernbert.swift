import Foundation;
import Darwin;

/**
 * Shallow discovery rules for executable ModernBERT embedding artifacts,
 * porting crates/config/src/model_discovery/modernbert.rs. The caseless
 * enum is the Swift equivalent of the Rust module of free functions.
 * Geometry and completeness prove an executable text-embedding artifact;
 * embedding inference is a single encoder forward pass, so discovery
 * derives a vector width and prompt budget instead of autoregressive token
 * limits.
 */
internal enum Modernbert {

    /** Family-derived metadata returned to neutral discovery orchestration. */
    internal struct DiscoveredModelMetadata: Equatable, Sendable {
        internal let vectorWidth: UInt32;
        internal let maximumInputTokens: UInt32;
        internal let modelSizeBytes: UInt64;

        internal init(
            vectorWidth: UInt32,
            maximumInputTokens: UInt32,
            modelSizeBytes: UInt64
        ) {
            self.vectorWidth = vectorWidth;
            self.maximumInputTokens = maximumInputTokens;
            self.modelSizeBytes = modelSizeBytes;
        }
    }

    /** Recognizes the MLX-converted ModernBERT embedding model type. */
    internal static func recognizesModelType(_ modelType: String?) -> Bool {
        guard let presentModelType: String = modelType else {
            return false;
        }
        return presentModelType == "modernbert";
    }

    /**
     * Validates shallow ModernBERT completeness and derives public discovery
     * metadata. Nil means the directory is not a complete executable
     * text-embedding artifact.
     */
    internal static func discoverModelMetadata(
        modelDirectory: FilePath,
        configObject: Dictionary<String, Any>
    ) -> Modernbert.DiscoveredModelMetadata? {
        if !DiscoveryPathNavigation.isExistingRegularFile(path: modelDirectory.appending(component: "model.safetensors"))
            || !DiscoveryPathNavigation.isExistingRegularFile(path: modelDirectory.appending(component: "tokenizer.json"))
        {
            return nil;
        }
        guard let hiddenSizeValue: Any = configObject["hidden_size"] else {
            return nil;
        }
        guard let hiddenSizeRaw: UInt64 = Modernbert.unsignedInteger64Value(of: hiddenSizeValue) else {
            return nil;
        }
        // Rust's `as u32` truncating cast, kept faithfully: a wider declared
        // hidden size contributes only its low 32 bits.
        let vectorWidth: UInt32 = UInt32(truncatingIfNeeded: hiddenSizeRaw);
        if (vectorWidth == 0) {
            return nil;
        }
        guard let maximumPositionEmbeddingsValue: Any = configObject["max_position_embeddings"] else {
            return nil;
        }
        guard let maximumPositionEmbeddingsRaw: UInt64 = Modernbert.unsignedInteger64Value(of: maximumPositionEmbeddingsValue) else {
            return nil;
        }
        let maximumInputTokens: UInt32 = UInt32(truncatingIfNeeded: maximumPositionEmbeddingsRaw);
        if (maximumInputTokens < 2) {
            return nil;
        }
        guard let quantizationValue: Any = configObject["quantization"] else {
            return nil;
        }
        guard let quantizationObject: Dictionary<String, Any> = quantizationValue as? Dictionary<String, Any> else {
            return nil;
        }
        guard let bitsValue: Any = quantizationObject["bits"] else {
            return nil;
        }
        guard let bits: UInt64 = Modernbert.unsignedInteger64Value(of: bitsValue) else {
            return nil;
        }
        if (bits != 8) {
            // Only the reviewed affine 8-bit profile is executable; other
            // widths stay undiscoverable until an engine validates them.
            return nil;
        }
        guard let modelSizeBytes: UInt64 = Modernbert.measureModelSafetensorsBytes(modelDirectory: modelDirectory) else {
            return nil;
        }
        return Modernbert.DiscoveredModelMetadata(
            vectorWidth: vectorWidth,
            maximumInputTokens: maximumInputTokens,
            modelSizeBytes: modelSizeBytes
        );
    }

    private static func measureModelSafetensorsBytes(modelDirectory: FilePath) -> UInt64? {
        return Modernbert.regularFileSizeBytes(path: modelDirectory.appending(component: "model.safetensors"));
    }

    /** `fs::metadata(path).len()`: symlink-following file size in bytes. */
    private static func regularFileSizeBytes(path: FilePath) -> UInt64? {
        var pathStatus: stat = stat();
        guard Darwin.fstatat(Darwin.AT_FDCWD, path.string, &pathStatus, 0) == 0 else {
            return nil;
        }
        return UInt64(pathStatus.st_size);
    }

    /**
     * serde_json's `Value::as_u64` over the JSONSerialization-shaped tree:
     * only numbers stored as unsigned integers qualify, so a whole-valued
     * float, a negative integer, a boolean, and a non-number all miss.
     */
    private static func unsignedInteger64Value(of jsonValue: Any) -> UInt64? {
        if (jsonValue is NSNull) {
            return nil;
        }
        guard let numberValue: NSNumber = jsonValue as? NSNumber else {
            return nil;
        }
        if CFGetTypeID(numberValue as CFTypeRef) == CFBooleanGetTypeID() {
            return nil;
        }
        let storageTypeDescription: String = String(cString: numberValue.objCType);
        if storageTypeDescription == "d" || storageTypeDescription == "f" {
            return nil;
        }
        if storageTypeDescription == "Q" {
            return numberValue.uint64Value;
        }
        let storedSignedValue: Int64 = numberValue.int64Value;
        guard storedSignedValue >= 0 else {
            return nil;
        }
        return UInt64(storedSignedValue);
    }
}
