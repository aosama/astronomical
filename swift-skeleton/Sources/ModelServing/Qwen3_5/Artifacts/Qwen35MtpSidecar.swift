import Foundation;

/// Validated Qwen-specific declaration from `mlx_lm_extra_tensors.mtp_file`.
/// Port of crates/model-serving/src/qwen3_5/artifacts/sidecar_declaration.rs.
public struct Qwen35MtpSidecarDeclaration: Equatable, Sendable {
    /// The one Qwen architecture sidecar receives a reserved opaque identity.
    /// Main-index sources are assigned upward from one, and the bounded index
    /// document cannot contain enough file names to reach this value, so the
    /// source cannot alias an indexed physical file.
    public static let MTP_SIDECAR_SOURCE_ID: TensorSourceId =
        TensorSourceId(sourceNumber: UInt32.max);
    public static let MTP_CANONICAL_PREFIX: String = "language_model.mtp.";
    static let MTP_STORED_PREFIX: String = "mtp.";
    private static let MAXIMUM_SIDECAR_RELATIVE_PATH_BYTES: Int = 4_096;

    private let relativePathValue: String;

    private init(relativePath: String) {
        self.relativePathValue = relativePath;
    }

    /// Validates declaration syntax before any filesystem operation.
    public static func parse(relativePath: String) throws -> Qwen35MtpSidecarDeclaration {
        let fileComponent: String = (relativePath as NSString).lastPathComponent;
        var hasAmbiguousTextComponent: Bool = false;
        for pathComponent in relativePath.split(separator: "/", omittingEmptySubsequences: false) {
            let componentName: String = String(pathComponent);
            if componentName.isEmpty || componentName == "." || componentName == ".." {
                hasAmbiguousTextComponent = true;
                break;
            }
        }
        let endsWithColonComponent: Bool = relativePath
            .split(separator: "/", omittingEmptySubsequences: false)
            .contains(where: { (pathComponent: Substring) -> Bool in
                return String(pathComponent).hasSuffix(":");
            });
        // Rust's `Path::extension` sees no extension on a leading-dot file name,
        // so a bare ".safetensors" component is not a declared sidecar file.
        let hasSafetensorsSuffix: Bool = fileComponent.hasPrefix(".") == false
            && (fileComponent as NSString).pathExtension == "safetensors";
        if relativePath.isEmpty
            || relativePath.utf8.count > Qwen35MtpSidecarDeclaration.MAXIMUM_SIDECAR_RELATIVE_PATH_BYTES
            || relativePath.hasPrefix("/")
            || relativePath.hasSuffix("/")
            || hasAmbiguousTextComponent
            || hasSafetensorsSuffix == false
            || relativePath.contains("\\")
            || endsWithColonComponent {
            throw Qwen35MtpSidecarDeclarationError.unsafeRelativePath;
        }
        return Qwen35MtpSidecarDeclaration(relativePath: relativePath);
    }

    public var relativePath: String {
        return self.relativePathValue;
    }
}

/// Bounded declaration failure that never includes a local path.
public enum Qwen35MtpSidecarDeclarationError: Error, Equatable, Sendable {
    case unsafeRelativePath;

    public var errorDescription: String? {
        switch self {
        case .unsafeRelativePath:
            return "MTP sidecar declaration must be a bounded relative SafeTensors path";
        }
    }
}

/// Structured validation failure for an optional Qwen MTP sidecar. Replaces
/// the earlier unit error so `mtp_unavailable_reason` can surface a
/// human-readable cause (for example which tensor is missing or which dtype
/// mismatched) instead of a silent target-only fallback.
public enum Qwen35MtpSidecarValidationError: Error, Equatable, Sendable {
    case sidecarFileUnavailable(relativePath: String);
    case unknownStoredTensor(tensorName: String);
    case duplicateCanonicalTensor(canonicalName: String);
    case targetTensorCollision(canonicalName: String);
    case missingProfileTensor(tensorName: String);
    case inventoryConflict(canonicalName: String);
    case profileValidationFailed(tensorName: String, detail: String);

    public var errorDescription: String? {
        switch self {
        case .sidecarFileUnavailable(let relativePath):
            return "MTP sidecar file '\(relativePath)' is unavailable or unparseable";
        case .unknownStoredTensor(let tensorName):
            return "sidecar tensor name '\(tensorName)' does not use the 'mtp.' stored prefix";
        case .duplicateCanonicalTensor(let canonicalName):
            return "sidecar declares duplicate canonical tensor '\(canonicalName)'";
        case .targetTensorCollision(let canonicalName):
            return "sidecar tensor '\(canonicalName)' collides with a target tensor";
        case .missingProfileTensor(let tensorName):
            return "expected tensor \(tensorName) not found in sidecar";
        case .inventoryConflict(let canonicalName):
            return "sidecar tensor inventory conflict for '\(canonicalName)'";
        case .profileValidationFailed(_, let detail):
            return detail;
        }
    }
}

/// Validated optional sidecar ownership transferred into the complete artifact.
public struct ValidatedQwen35MtpSidecar {
    let source: ValidatedSafetensorsSource;
    let inventory: TensorInventory;
}

/// Hermetic optional-sidecar validation outcome.
public struct Qwen35MtpSidecarValidationOutcome {
    private let inventory: TensorInventory;
    private let payloadBytesValue: UInt64;
    private let availabilityFlag: Bool;

    init(inventory: TensorInventory, payloadBytes: UInt64, isAvailable: Bool) {
        self.inventory = inventory;
        self.payloadBytesValue = payloadBytes;
        self.availabilityFlag = isAvailable;
    }

    /// The unavailable default: no sidecar tensors, no payload.
    public static func unavailable() -> Qwen35MtpSidecarValidationOutcome {
        return Qwen35MtpSidecarValidationOutcome(
            inventory: TensorInventory(), payloadBytes: 0, isAvailable: false);
    }

    public func isAvailable() -> Bool {
        return self.availabilityFlag;
    }

    public func sourceCount() -> Int {
        return self.inventory.sourceIds().count;
    }

    public func tensorCount() -> Int {
        return self.inventory.tensorCount();
    }

    public func payloadBytes() -> UInt64 {
        return self.payloadBytesValue;
    }

    public func storedName(canonicalName: String) -> String? {
        return self.inventory.location(canonicalName: canonicalName)?.storedName;
    }
}

/// Already-open optional Qwen MTP source and its bounded header name mapping.
public final class Qwen35MtpSidecarCandidate {
    private let source: ValidatedSafetensorsSource;
    /// Canonical-to-stored mapping in canonical-name order, mirroring the
    /// Rust B-tree iteration.
    private let canonicalToStoredName: Dictionary<String, String>;

    private init(
        source: ValidatedSafetensorsSource,
        canonicalToStoredName: Dictionary<String, String>) {
        self.source = source;
        self.canonicalToStoredName = canonicalToStoredName;
    }

    public static func open(
        modelDirectory: String,
        declaration: Qwen35MtpSidecarDeclaration) throws -> Qwen35MtpSidecarCandidate {
        let requiredFile: ValidatedRequiredFile;
        do {
            requiredFile = try RequiredFiles.validateRequiredFile(
                modelDirectory: modelDirectory,
                requiredFileProfile: RequiredFileProfile(
                    fileName: declaration.relativePath, sizeBytes: 0));
        } catch {
            throw Qwen35MtpSidecarValidationError.sidecarFileUnavailable(
                relativePath: declaration.relativePath);
        }
        let source: ValidatedSafetensorsSource;
        do {
            source = try ValidatedSafetensorsSource.parse(
                sourceId: Qwen35MtpSidecarDeclaration.MTP_SIDECAR_SOURCE_ID,
                requiredFile: requiredFile);
        } catch {
            throw Qwen35MtpSidecarValidationError.sidecarFileUnavailable(
                relativePath: declaration.relativePath);
        }
        var canonicalToStoredName: Dictionary<String, String> = Dictionary();
        for storedName: String in source.storedTensorNames() {
            guard storedName.hasPrefix(Qwen35MtpSidecarDeclaration.MTP_STORED_PREFIX) else {
                throw Qwen35MtpSidecarValidationError.unknownStoredTensor(tensorName: storedName);
            }
            let suffix: String = String(
                storedName.dropFirst(Qwen35MtpSidecarDeclaration.MTP_STORED_PREFIX.count));
            let canonicalName: String =
                Qwen35MtpSidecarDeclaration.MTP_CANONICAL_PREFIX + suffix;
            if canonicalToStoredName[canonicalName] != nil {
                throw Qwen35MtpSidecarValidationError.duplicateCanonicalTensor(
                    canonicalName: canonicalName);
            }
            canonicalToStoredName[canonicalName] = storedName;
        }
        return Qwen35MtpSidecarCandidate(
            source: source, canonicalToStoredName: canonicalToStoredName);
    }

    public func canonicalNames() -> Array<String> {
        return self.canonicalToStoredName.keys.sorted();
    }

    /// Validates the sidecar against the generated profile set and the
    /// already-declared main-index canonical names.
    public func validate(
        canonicalProfiles: Array<TensorProfile>,
        existingCanonicalNames: Array<String>) throws -> ValidatedQwen35MtpSidecar {
        var profileByCanonicalName: Dictionary<String, TensorProfile> = Dictionary();
        for tensorProfile: TensorProfile in canonicalProfiles {
            profileByCanonicalName[tensorProfile.name] = tensorProfile;
        }
        // An optional sidecar may carry tensors beyond the generated profile
        // set (for example a quantized head that enumerates per-expert
        // weights). Such additional tensors are accepted as future
        // extensibility; only the canonical names described by profiles must
        // validate.
        let existingNames: Set<String> = Set(existingCanonicalNames);
        let inventory: TensorInventory = TensorInventory();
        for canonicalName: String in self.canonicalToStoredName.keys.sorted() {
            guard let storedName: String = self.canonicalToStoredName[canonicalName] else {
                preconditionFailure(
                    "canonical mapping must contain every name it was built from");
            }
            if existingNames.contains(canonicalName) {
                throw Qwen35MtpSidecarValidationError.targetTensorCollision(
                    canonicalName: canonicalName);
            }
            guard let tensorProfile: TensorProfile = profileByCanonicalName[canonicalName] else {
                continue;
            }
            guard let metadata: SafetensorsFraming.TensorView =
                self.source.storedTensorView(storedName: storedName) else {
                preconditionFailure(
                    "stored tensor metadata must exist after a successful sidecar parse");
            }
            do {
                try Qwen35MtpSidecarCandidate.validateTensorProfile(
                    profile: tensorProfile, metadata: metadata);
            } catch let profileDetail as Qwen35MtpSidecarProfileDetail {
                throw Qwen35MtpSidecarValidationError.profileValidationFailed(
                    tensorName: canonicalName, detail: profileDetail.detailText);
            }
            do {
                try inventory.insert(location: TensorLocation(
                    canonicalName: canonicalName, storedName: storedName,
                    sourceId: self.source.sourceId,
                    semanticRole: .multiTokenPrediction,
                    declarationOrigin: .architectureSidecar,
                    feature: .multiTokenPrediction));
            } catch {
                throw Qwen35MtpSidecarValidationError.inventoryConflict(
                    canonicalName: canonicalName);
            }
        }
        // Every generated profile must have a matching stored tensor in the sidecar.
        for tensorProfile: TensorProfile in canonicalProfiles {
            if self.canonicalToStoredName[tensorProfile.name] == nil {
                throw Qwen35MtpSidecarValidationError.missingProfileTensor(
                    tensorName: tensorProfile.name);
            }
        }
        return ValidatedQwen35MtpSidecar(source: self.source, inventory: inventory);
    }

    /// Validates one tensor profile against the sidecar's retained header
    /// metadata, producing a human-readable detail on mismatch.
    private static func validateTensorProfile(
        profile: TensorProfile, metadata: SafetensorsFraming.TensorView) throws -> Void {
        let expectedDtypeText: String;
        switch profile.dtype {
        case .affineQuantizationFloat, .modelFloat:
            expectedDtypeText = "float (F16/BF16/F32)";
        case .bfloat16:
            expectedDtypeText = "BF16";
        case .float32:
            expectedDtypeText = "F32";
        case .uint32:
            expectedDtypeText = "U32";
        }
        let acceptedDtypeNames: Set<String>;
        switch profile.dtype {
        case .affineQuantizationFloat, .modelFloat:
            acceptedDtypeNames = ["F16", "BF16", "F32"];
        case .bfloat16:
            acceptedDtypeNames = ["BF16"];
        case .float32:
            acceptedDtypeNames = ["F32"];
        case .uint32:
            acceptedDtypeNames = ["U32"];
        }
        if acceptedDtypeNames.contains(metadata.dtype) == false {
            throw Qwen35MtpSidecarProfileDetail(detailText:
                "dtype mismatch: expected \(expectedDtypeText), got \(metadata.dtype) "
                    + "for tensor \(profile.name)");
        }
        if metadata.shape != profile.shape {
            throw Qwen35MtpSidecarProfileDetail(detailText:
                "shape mismatch: expected \(Qwen35MtpSidecarCandidate.rustDebugArrayText(profile.shape)), "
                    + "got \(Qwen35MtpSidecarCandidate.rustDebugArrayText(metadata.shape)) "
                    + "for tensor \(profile.name)");
        }
    }

    /// Reproduces Rust's `{:?}` slice rendering, e.g. `[2, 3]`.
    private static func rustDebugArrayText(_ values: Array<Int>) -> String {
        return "[\(values.map({ (value: Int) -> String in String(value) }).joined(separator: ", "))]";
    }
}

/// Internal carrier for the human-readable profile mismatch detail the sidecar
/// error wraps, mirroring the Rust `Result<(), String>` detail.
private struct Qwen35MtpSidecarProfileDetail: Error {
    let detailText: String;
}

/// Optional-sidecar validation entry points. The sidecar never rejects target
/// serving: failures degrade to the unavailable outcome on the lenient seam.
public enum Qwen35MtpSidecar {

    /// Surface the structured validation outcome for hermetic tests that
    /// assert diagnostics.
    public static func validateResultForTests(
        modelDirectory: String, declaration: Qwen35MtpSidecarDeclaration,
        canonicalProfiles: Array<TensorProfile>,
        existingCanonicalNames: Array<String>) throws -> Qwen35MtpSidecarValidationOutcome {
        return try Qwen35MtpSidecar.validateOptionalSidecar(
            modelDirectory: modelDirectory, declaration: declaration,
            canonicalProfiles: canonicalProfiles,
            existingCanonicalNames: existingCanonicalNames);
    }

    /// Hermetic seam returning the unavailable default on any failure.
    public static func validateForTests(
        modelDirectory: String, declaration: Qwen35MtpSidecarDeclaration,
        canonicalProfiles: Array<TensorProfile>,
        existingCanonicalNames: Array<String>) -> Qwen35MtpSidecarValidationOutcome {
        do {
            return try Qwen35MtpSidecar.validateOptionalSidecar(
                modelDirectory: modelDirectory, declaration: declaration,
                canonicalProfiles: canonicalProfiles,
                existingCanonicalNames: existingCanonicalNames);
        } catch {
            return Qwen35MtpSidecarValidationOutcome.unavailable();
        }
    }

    private static func validateOptionalSidecar(
        modelDirectory: String, declaration: Qwen35MtpSidecarDeclaration,
        canonicalProfiles: Array<TensorProfile>,
        existingCanonicalNames: Array<String>) throws -> Qwen35MtpSidecarValidationOutcome {
        let validatedSidecar: ValidatedQwen35MtpSidecar = try Qwen35MtpSidecarCandidate
            .open(modelDirectory: modelDirectory, declaration: declaration)
            .validate(
                canonicalProfiles: canonicalProfiles,
                existingCanonicalNames: existingCanonicalNames);
        return Qwen35MtpSidecarValidationOutcome(
            inventory: validatedSidecar.inventory,
            payloadBytes: validatedSidecar.source.payloadBytes,
            isAvailable: true);
    }
}
