import Foundation

import AstronomicalConfig

/// Publication and adoption of a downloaded model under the models
/// directory: provenance evidence both this coordinator and model discovery
/// trust, plus the pre-existing-publication reconciliation rules.
public enum LibraryDownloadPublication {

    public static let provenanceFileName: String = ".astronomical-library-provenance.json"

    /// The published location for one Hugging Face identity:
    /// `<modelsDirectory>/<organization>/<name>`.
    public static func destinationDirectory(
        huggingfaceId: String,
        modelsDirectory: FilePath
    ) -> FilePath {
        let identityComponents: [Substring] = huggingfaceId.split(separator: "/")
        let organization: Substring = identityComponents.first ?? ""
        let modelName: Substring = identityComponents.count > 1 ? identityComponents[1] : ""
        return modelsDirectory
            .appending(component: String(organization))
            .appending(component: String(modelName))
    }

    /// Provenance evidence for an existing publication, or nil when the
    /// destination has none.
    public static func existingProvenance(destinationDirectory: FilePath) -> (providerModelId: String, revision: String)? {
        let provenanceFilePath: FilePath = destinationDirectory.appending(
            component: LibraryDownloadPublication.provenanceFileName)
        guard let provenanceBytes: Data = FileManager.default.contents(atPath: provenanceFilePath.string) else {
            return nil
        }
        guard let provenanceObject: [String: Any] = try? JSONSerialization.jsonObject(with: provenanceBytes) as? [String: Any] else {
            return nil
        }
        guard let schemaVersion: Int = provenanceObject["schema_version"] as? Int,
            schemaVersion == 1,
            let providerModelId: String = provenanceObject["provider_model_id"] as? String,
            let revision: String = provenanceObject["revision"] as? String
        else {
            return nil
        }
        return (providerModelId: providerModelId, revision: revision)
    }

    /// Writes the durable provenance pair discovery reads: the library
    /// provenance document plus the hidden Hub revision marker. Existing
    /// files are left untouched so adoption never rewrites a byte of a
    /// pre-existing publication.
    public static func writeProvenance(
        destinationDirectory: FilePath,
        providerModelId: String,
        revision: String
    ) throws {
        let provenanceDocument: [String: Any] = [
            "schema_version": 1,
            "provider_model_id": providerModelId,
            "revision": revision,
        ]
        let provenanceBytes: Data = try JSONSerialization.data(
            withJSONObject: provenanceDocument,
            options: [.sortedKeys])
        try FileManager.default.createDirectory(
            atPath: destinationDirectory.string,
            withIntermediateDirectories: true)
        try provenanceBytes.write(
            to: URL(fileURLWithPath: destinationDirectory.appending(
                component: LibraryDownloadPublication.provenanceFileName).string),
            options: [.atomic])
        let revisionMarkerDirectory: FilePath = destinationDirectory
            .appending(component: ".cache")
            .appending(component: "huggingface")
            .appending(component: "download")
        try FileManager.default.createDirectory(
            atPath: revisionMarkerDirectory.string,
            withIntermediateDirectories: true)
        try Data("\(revision)\n".utf8).write(
            to: URL(fileURLWithPath: revisionMarkerDirectory.appending(
                component: "config.json.metadata").string),
            options: [.atomic])
    }
}
