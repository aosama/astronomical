import Foundation;
import Darwin;

/**
 * Minimal ports of classified_artifacts.rs provenance helpers for Flux2Klein
 * discovery: they decide whether the model directory carries trustworthy
 * immutable identity (library provenance file or Hub download metadata).
 */
internal enum Flux2KleinProvenance {
    private static let MAXIMUM_REVISION_METADATA_BYTES: UInt64 = 4_096;
    private static let MAXIMUM_LIBRARY_PROVENANCE_BYTES: UInt64 = 1_024;
    private static let LIBRARY_PROVENANCE_FILE_NAME: String = ".astronomical-library-provenance.json";

    /**
     * Returns (provider_model_id, revision) recorded by the packaging library
     * next to the model, or nil when no trustworthy provenance file exists.
     */
    internal static func immutableModelProvenance(modelDirectory: FilePath) -> (providerModelId: String, revision: String)? {
        let provenanceFilePath: FilePath = modelDirectory.appending(component: self.LIBRARY_PROVENANCE_FILE_NAME);
        var provenanceStatus: stat = stat();
        // lstat: a symlinked provenance file is not trusted evidence.
        guard Darwin.fstatat(Darwin.AT_FDCWD, provenanceFilePath.string, &provenanceStatus, Darwin.AT_SYMLINK_NOFOLLOW) == 0 else {
            return nil;
        }
        guard (provenanceStatus.st_mode & S_IFMT) == S_IFREG else {
            return nil;
        }
        guard UInt64(provenanceStatus.st_size) <= self.MAXIMUM_LIBRARY_PROVENANCE_BYTES else {
            return nil;
        }
        guard let provenanceBytes: Data = FileManager.default.contents(atPath: provenanceFilePath.string) else {
            return nil;
        }
        let provenance: LibraryModelProvenance;
        do {
            let parsedDocumentValue: Any = try DiscoveryStrictJsonDocument.parseDocument(bytes: provenanceBytes);
            guard let provenanceObject: Dictionary<String, Any> = parsedDocumentValue as? Dictionary<String, Any> else {
                return nil;
            }
            provenance = try LibraryModelProvenance.fromJsonObject(provenanceObject);
        } catch {
            return nil;
        }
        guard provenance.schemaVersion == 1 else {
            return nil;
        }
        guard self.isLowercaseHexRevision(revision: provenance.revision) else {
            return nil;
        }
        guard self.isValidProviderModelId(providerModelId: provenance.providerModelId) else {
            return nil;
        }
        return (providerModelId: provenance.providerModelId, revision: provenance.revision);
    }

    private static func isValidProviderModelId(providerModelId: String) -> Bool {
        let identityComponents: Array<String> = providerModelId.split(
            omittingEmptySubsequences: false,
            whereSeparator: { (pathCharacter: Character) -> Bool in return pathCharacter == "/"; }
        ).map({ (pathComponent: Substring) -> String in return String(pathComponent); });
        guard identityComponents.count == 2 else {
            return false;
        }
        for identityComponent: String in identityComponents {
            if identityComponent.isEmpty || identityComponent == "." || identityComponent == ".." {
                return false;
            }
            for componentByte: UInt8 in identityComponent.utf8 {
                let isAsciiDigit: Bool = componentByte >= 0x30 && componentByte <= 0x39;
                let isAsciiUppercase: Bool = componentByte >= 0x41 && componentByte <= 0x5A;
                let isAsciiLowercase: Bool = componentByte >= 0x61 && componentByte <= 0x7A;
                let isAllowedPunctuation: Bool = componentByte == 0x2D || componentByte == 0x5F || componentByte == 0x2E;
                if !isAsciiDigit && !isAsciiUppercase && !isAsciiLowercase && !isAllowedPunctuation {
                    return false;
                }
            }
        }
        return true;
    }

    private static func isLowercaseHexRevision(revision: String) -> Bool {
        guard revision.utf8.count == 40 else {
            return false;
        }
        for revisionByte: UInt8 in revision.utf8 {
            let isAsciiDigit: Bool = revisionByte >= 0x30 && revisionByte <= 0x39;
            let isLowercaseHexDigit: Bool = revisionByte >= 0x61 && revisionByte <= 0x66;
            if !isAsciiDigit && !isLowercaseHexDigit {
                return false;
            }
        }
        return true;
    }

    /**
     * Returns the recorded first line of the Hub download metadata for the
     * authoritative file, or the model directory's cache-decoded leaf name
     * when the directory itself is a Hub cache snapshot; nil otherwise.
     */
    internal static func immutableFileRevision(modelDirectory: FilePath, authoritativeFileName: String) -> String? {
        let localMetadataPath: FilePath = modelDirectory.appending(
            component: ".cache/huggingface/download/" + authoritativeFileName + ".metadata"
        );
        var metadataStatus: stat = stat();
        let hasBoundedLocalMetadata: Bool = Darwin.fstatat(Darwin.AT_FDCWD, localMetadataPath.string, &metadataStatus, 0) == 0
            && UInt64(metadataStatus.st_size) <= self.MAXIMUM_REVISION_METADATA_BYTES;
        if (hasBoundedLocalMetadata) {
            // Metadata read failures fall through to nil; the ancestor cache
            // fallback only applies when no bounded local metadata exists.
            guard let metadataBytes: Data = FileManager.default.contents(atPath: localMetadataPath.string) else {
                return nil;
            }
            guard let metadataText: String = String(data: metadataBytes, encoding: .utf8) else {
                return nil;
            }
            let firstLineText: String;
            if let firstNewlineIndex: String.Index = metadataText.firstIndex(of: "\n") {
                var lineWithoutNewline: String = String(metadataText[metadataText.startIndex..<firstNewlineIndex]);
                if (lineWithoutNewline.hasSuffix("\r")) {
                    lineWithoutNewline.removeLast();
                }
                firstLineText = lineWithoutNewline;
            } else {
                firstLineText = metadataText;
            }
            guard !firstLineText.isEmpty else {
                return nil;
            }
            return firstLineText;
        }
        for ancestorPath: FilePath in DiscoveryPathNavigation.ancestorDirectoryPaths(startingFrom: modelDirectory) {
            guard let ancestorName: String = DiscoveryPathNavigation.lastComponentName(of: ancestorPath) else {
                continue;
            }
            if (DiscoveryHuggingFaceCache.decodeCacheDirectoryName(directoryName: ancestorName) != nil) {
                guard let modelDirectoryName: String = DiscoveryPathNavigation.lastComponentName(of: modelDirectory) else {
                    return nil;
                }
                return modelDirectoryName;
            }
        }
        return nil;
    }

    /** Wire shape of `.astronomical-library-provenance.json` (deny_unknown_fields in Rust). */
    private struct LibraryModelProvenance {
        fileprivate let schemaVersion: UInt32;
        fileprivate let providerModelId: String;
        fileprivate let revision: String;

        fileprivate init(schemaVersion: UInt32, providerModelId: String, revision: String) {
            self.schemaVersion = schemaVersion;
            self.providerModelId = providerModelId;
            self.revision = revision;
        }

        fileprivate static func fromJsonObject(_ jsonObject: Dictionary<String, Any>) throws -> Flux2KleinProvenance.LibraryModelProvenance {
            try StrictJson.requireKnownKeys(
                object: jsonObject,
                knownKeys: Set<String>(["schema_version", "provider_model_id", "revision"]),
                fieldName: ""
            );
            return Flux2KleinProvenance.LibraryModelProvenance(
                schemaVersion: try StrictJson.requiredUnsignedInteger(object: jsonObject, fieldName: "schema_version"),
                providerModelId: try StrictJson.requiredString(object: jsonObject, fieldName: "provider_model_id"),
                revision: try StrictJson.requiredString(object: jsonObject, fieldName: "revision")
            );
        }
    }
}
