import Foundation;
import IpcProtocol;

/// Per-module OptiQ quantization types, port of
/// crates/model-serving/src/qwen3_5/quantizations/optiq/config.rs.
enum OptiQQuantizationConstants {
    static let EXPECTED_QUANTIZATION_MODE: String = "affine";
}

/// Returns whether MLX supports the affine quantization bit width.
public func isMlxAffineQuantizationBitWidthSupported(bitWidth: UInt32) -> Bool {
    switch bitWidth {
    case 2, 3, 4, 5, 6, 8: return true;
    default: return false;
    }
}

/// Returns whether MLX supports the affine quantization group size.
public func isMlxAffineQuantizationGroupSizeSupported(groupSize: UInt32) -> Bool {
    switch groupSize {
    case 32, 64, 128: return true;
    default: return false;
    }
}

/// Per-module quantization profile carrying both bit width and group size.
///
/// Each quantized module in a Qwen3.5 model can have its own bit width
/// and group size. The OptiQ mixed-precision format uses sparse overrides:
/// modules not listed in the config use the default `(bits, group_size)`.
///
/// A profile with `bits = 0` indicates an unquantized module stored as
/// bfloat16 (no scales/biases tensors on disk). This is used for the
/// Some artifacts use this for native floating-point modules inside an otherwise
/// affine-quantized model.
public struct OptiQQuantizationProfile: Equatable {
    public let bits: UInt32;
    public let groupSize: UInt32;

    public init(bits: UInt32, groupSize: UInt32) {
        self.bits = bits;
        self.groupSize = groupSize;
    }

    /// Returns true when the module is stored as plain bfloat16 (no quantization).
    ///
    /// Unquantized modules have `bits = 0` and no `.scales`/`.biases` tensors
    /// on disk. The weight is loaded directly as a bfloat16 tensor.
    public func isUnquantized() -> Bool {
        return self.bits == 0;
    }

    /// Creates an unquantized profile (bits=0, group_size=0) indicating bfloat16 storage.
    public static func unquantized() -> OptiQQuantizationProfile {
        return OptiQQuantizationProfile(bits: 0, groupSize: 0);
    }
}

/// Text config fields needed by the quantization validator.
public protocol QuantizationConfigSource {
    func layerCount() -> UInt32;
    func decoderLayerIsFullAttention(decoderLayerIndex: Int) -> Bool;
}

/// The two identical quantization documents embedded in the Qwen3.5 config.
///
/// Supports both the original OptiQ-4bit format (explicit overrides for every
/// module) and the newer mixed-precision format (sparse overrides where modules
/// not listed use the default `bits` and `group_size`).
public struct OptiQQuantizationConfig: Equatable {
    private let groupSizeValue: UInt32;
    private let bitsValue: UInt32;
    private let modeName: String;
    /// Module overrides collected from the flattened unknown keys of the
    /// quantization document, kept in Rust BTreeMap (UTF-8 byte) order.
    private let moduleOverrides: Array<(moduleName: String, overrideValue: OptiQQuantizationOverride)>;

    fileprivate init(
        groupSize: UInt32, bits: UInt32, mode: String,
        moduleOverrides: Array<(moduleName: String, overrideValue: OptiQQuantizationOverride)>) {
        self.groupSizeValue = groupSize;
        self.bitsValue = bits;
        self.modeName = mode;
        self.moduleOverrides = moduleOverrides;
    }

    /// Content equality: Rust's derived Eq compares the BTreeMap by content,
    /// not by iteration container identity.
    public static func == (lhsValue: OptiQQuantizationConfig, rhsValue: OptiQQuantizationConfig) -> Bool {
        return lhsValue.groupSizeValue == rhsValue.groupSizeValue
            && lhsValue.bitsValue == rhsValue.bitsValue
            && lhsValue.modeName == rhsValue.modeName
            && lhsValue.moduleOverrides.count == rhsValue.moduleOverrides.count
            && lhsValue.moduleOverrides.elementsEqual(rhsValue.moduleOverrides, by: { (leftEntry: (moduleName: String, overrideValue: OptiQQuantizationOverride), rightEntry: (moduleName: String, overrideValue: OptiQQuantizationOverride)) -> Bool in
                return leftEntry.moduleName == rightEntry.moduleName
                    && leftEntry.overrideValue == rightEntry.overrideValue;
            });
    }

    public static func decoded(wireValue: JsonWireValue) throws -> OptiQQuantizationConfig {
        let quantizationObject: JsonWireObject = try JsonWireValue.extractObject(wireValue);
        let knownFieldNames: Array<String> = ["bits", "group_size", "mode"];
        var moduleOverrides: Array<(moduleName: String, overrideValue: OptiQQuantizationOverride)> = Array();
        for propertyName: String in quantizationObject.keyNames {
            if knownFieldNames.contains(propertyName) {
                continue;
            }
            let overrideObject: JsonWireObject = try JsonWireValue.extractObject(
                try quantizationObject.requireObjectValue(fieldName: propertyName));
            try overrideObject.rejectUnknownFields(allowedFieldNames: ["bits", "group_size"]);
            moduleOverrides.append((
                propertyName,
                OptiQQuantizationOverride(
                    groupSize: try overrideObject.decodeUInt32(fieldName: "group_size"),
                    bits: try overrideObject.decodeUInt32(fieldName: "bits"))));
        }
        // Rust collects the flattened overrides in a BTreeMap, so iteration
        // order follows UTF-8 byte ordering.
        moduleOverrides.sort { (leftEntry: (moduleName: String, overrideValue: OptiQQuantizationOverride), rightEntry: (moduleName: String, overrideValue: OptiQQuantizationOverride)) -> Bool in
            return Array(leftEntry.moduleName.utf8).lexicographicallyPrecedes(Array(rightEntry.moduleName.utf8));
        };
        return OptiQQuantizationConfig(
            groupSize: try quantizationObject.decodeUInt32(fieldName: "group_size"),
            bits: try quantizationObject.decodeUInt32(fieldName: "bits"),
            mode: try quantizationObject.decodeString(fieldName: "mode"),
            moduleOverrides: moduleOverrides);
    }

    /// Returns the default quantization group size declared by the artifact.
    public func defaultGroupSize() -> UInt32 {
        return self.groupSizeValue;
    }

    /// Returns the default quantization bit width declared by the artifact.
    public func defaultBits() -> UInt32 {
        return self.bitsValue;
    }

    /// Validates the quantization config and returns a fully resolved map of
    /// module name to quantization profile (bits + group_size).
    ///
    /// For modules not listed in overrides, the default `(bits, group_size)`
    /// is used. Supports OptiQ-4bit (explicit all-module overrides with bits=4),
    /// and mixed-precision formats (sparse overrides with default bits=6).
    public func validate(
        configSource: QuantizationConfigSource,
        feedForwardArchitecture: Qwen3_5FeedForwardArchitecture) throws -> SortedModuleProfiles {
        try Qwen3_5ConfigValidation.validateExactValue(
            fieldName: "quantization.mode", actualValue: self.modeName,
            expectedValue: OptiQQuantizationConstants.EXPECTED_QUANTIZATION_MODE);
        if isMlxAffineQuantizationBitWidthSupported(bitWidth: self.bitsValue) == false {
            throw Qwen3_5ConfigError.invalidConfigValueDynamic(
                description: "quantization.bits must be supported by MLX affine quantization, got \(self.bitsValue)");
        }
        if isMlxAffineQuantizationGroupSizeSupported(groupSize: self.groupSizeValue) == false {
            throw Qwen3_5ConfigError.invalidConfigValueDynamic(
                description: "quantization.group_size must be supported by MLX affine quantization, got \(self.groupSizeValue)");
        }
        // Validate each override against the same affine parameters MLX accepts.
        for moduleOverride: (moduleName: String, overrideValue: OptiQQuantizationOverride) in self.moduleOverrides {
            if isMlxAffineQuantizationBitWidthSupported(bitWidth: moduleOverride.overrideValue.bits) == false {
                throw Qwen3_5ConfigError.unsupportedQuantizationOverrideBits(
                    moduleName: moduleOverride.moduleName, actualValue: moduleOverride.overrideValue.bits);
            }
            if isMlxAffineQuantizationGroupSizeSupported(groupSize: moduleOverride.overrideValue.groupSize) == false {
                throw Qwen3_5ConfigError.invalidConfigValueDynamic(
                    description: "quantization module override '\(moduleOverride.moduleName)' group_size is not supported by MLX affine quantization: \(moduleOverride.overrideValue.groupSize)");
            }
        }
        // Build the fully resolved module profile map.
        // For OptiQ-4bit (bits=4): all modules are explicit overrides.
        // For mixed-precision (bits=6): overrides specify modules that differ from default.
        let defaultProfile: OptiQQuantizationProfile = OptiQQuantizationProfile(
            bits: self.bitsValue, groupSize: self.groupSizeValue);
        let allModuleNames: Array<String> = Self.expectedQuantizedModuleNames(
            configSource: configSource, feedForwardArchitecture: feedForwardArchitecture);
        var resolvedProfiles: Array<(moduleName: String, profile: OptiQQuantizationProfile)> = Array();
        resolvedProfiles.reserveCapacity(allModuleNames.count);
        for moduleName: String in allModuleNames {
            let profile: OptiQQuantizationProfile;
            if let moduleOverride: (moduleName: String, overrideValue: OptiQQuantizationOverride)
                = self.moduleOverrides.first(where: { (entry: (moduleName: String, overrideValue: OptiQQuantizationOverride)) -> Bool in entry.moduleName == moduleName }) {
                profile = OptiQQuantizationProfile(
                    bits: moduleOverride.overrideValue.bits,
                    groupSize: moduleOverride.overrideValue.groupSize);
            } else {
                profile = defaultProfile;
            }
            resolvedProfiles.append((moduleName, profile));
        }
        return SortedModuleProfiles(entries: resolvedProfiles);
    }

    /// Returns the artifact-declared quantization profiles for optional MTP modules.
    public func mtpQuantizedModuleProfiles() -> SortedModuleProfiles {
        var mtpProfiles: Array<(moduleName: String, profile: OptiQQuantizationProfile)> = Array();
        for moduleOverride: (moduleName: String, overrideValue: OptiQQuantizationOverride) in self.moduleOverrides {
            if moduleOverride.moduleName.hasPrefix("language_model.mtp.") {
                mtpProfiles.append((
                    moduleOverride.moduleName,
                    OptiQQuantizationProfile(
                        bits: moduleOverride.overrideValue.bits,
                        groupSize: moduleOverride.overrideValue.groupSize)));
            }
        }
        return SortedModuleProfiles(entries: mtpProfiles);
    }

    private static func expectedQuantizedModuleNames(
        configSource: QuantizationConfigSource,
        feedForwardArchitecture: Qwen3_5FeedForwardArchitecture) -> Array<String> {
        let layerCount: Int = Int(configSource.layerCount());
        var quantizedModuleNames: Array<String> = Array();
        for decoderLayerIndex: Int in 0..<layerCount {
            let layerPrefix: String = "language_model.model.layers.\(decoderLayerIndex)";
            if configSource.decoderLayerIsFullAttention(decoderLayerIndex: decoderLayerIndex) {
                for projectionName in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                    quantizedModuleNames.append("\(layerPrefix).self_attn.\(projectionName)");
                }
            } else {
                for projectionName in ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"] {
                    quantizedModuleNames.append("\(layerPrefix).linear_attn.\(projectionName)");
                }
            }
            let mlpModuleSuffixes: Array<String>;
            switch feedForwardArchitecture {
            case .dense:
                mlpModuleSuffixes = ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"];
            case .mixtureOfExperts:
                mlpModuleSuffixes = [
                    "mlp.gate",
                    "mlp.switch_mlp.gate_proj",
                    "mlp.switch_mlp.up_proj",
                    "mlp.switch_mlp.down_proj",
                    "mlp.shared_expert.gate_proj",
                    "mlp.shared_expert.up_proj",
                    "mlp.shared_expert.down_proj",
                    "mlp.shared_expert_gate",
                ];
            }
            for moduleSuffix: String in mlpModuleSuffixes {
                quantizedModuleNames.append("\(layerPrefix).\(moduleSuffix)");
            }
        }
        quantizedModuleNames.append("language_model.model.embed_tokens");
        quantizedModuleNames.append("language_model.lm_head");
        return quantizedModuleNames;
    }
}

/// One flattened quantization override inside a module-name key.
struct OptiQQuantizationOverride: Equatable {
    let groupSize: UInt32;
    let bits: UInt32;
}

/// Module-name-keyed profiles kept in Rust BTreeMap (UTF-8 byte) order so
/// iteration and equality match the Rust contract exactly.
public struct SortedModuleProfiles: Equatable {
    public private(set) var entries: Array<(moduleName: String, profile: OptiQQuantizationProfile)>;

    public static func == (lhsValue: SortedModuleProfiles, rhsValue: SortedModuleProfiles) -> Bool {
        return lhsValue.entries.elementsEqual(rhsValue.entries, by: { (leftEntry: (moduleName: String, profile: OptiQQuantizationProfile), rightEntry: (moduleName: String, profile: OptiQQuantizationProfile)) -> Bool in
            return leftEntry.moduleName == rightEntry.moduleName && leftEntry.profile == rightEntry.profile;
        });
    }

    public init(entries: Array<(moduleName: String, profile: OptiQQuantizationProfile)>) {
        var sortedEntries: Array<(moduleName: String, profile: OptiQQuantizationProfile)> = entries;
        sortedEntries.sort { (leftEntry: (moduleName: String, profile: OptiQQuantizationProfile), rightEntry: (moduleName: String, profile: OptiQQuantizationProfile)) -> Bool in
            return Array(leftEntry.moduleName.utf8).lexicographicallyPrecedes(Array(rightEntry.moduleName.utf8));
        };
        self.entries = sortedEntries;
    }

    public subscript(moduleName: String) -> OptiQQuantizationProfile? {
        return self.entries.first(where: { (entry: (moduleName: String, profile: OptiQQuantizationProfile)) -> Bool in entry.moduleName == moduleName })?.profile;
    }

    public func profile(forKey moduleName: String) -> OptiQQuantizationProfile? {
        return self[moduleName];
    }

    public var count: Int {
        return self.entries.count;
    }

    public mutating func insert(profile: OptiQQuantizationProfile, forKey moduleName: String) {
        if let existingIndex: Int = self.entries.firstIndex(where: { (entry: (moduleName: String, profile: OptiQQuantizationProfile)) -> Bool in entry.moduleName == moduleName }) {
            self.entries[existingIndex].profile = profile;
            return;
        }
        self.entries.append((moduleName, profile));
        self.entries.sort { (leftEntry: (moduleName: String, profile: OptiQQuantizationProfile), rightEntry: (moduleName: String, profile: OptiQQuantizationProfile)) -> Bool in
            return Array(leftEntry.moduleName.utf8).lexicographicallyPrecedes(Array(rightEntry.moduleName.utf8));
        };
    }

    /// Keeps only the entries the predicate accepts, mirroring BTreeMap
    /// retention filters on the Rust side.
    public mutating func filter(keeping acceptedEntry: ((moduleName: String, profile: OptiQQuantizationProfile)) -> Bool) {
        self.entries = self.entries.filter(acceptedEntry);
    }
}
