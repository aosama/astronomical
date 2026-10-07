import Foundation

import IpcProtocol

/**
 * Read-only REST projection of the validated release catalog. The response
 * shape mirrors the Rust catalog endpoint's serde document exactly: fields
 * with serde `skip_serializing_if` are omitted when absent, and
 * `download_state` is always present (null when no job is in flight).
 */
public enum LibraryCatalogEndpoint {

    public static func catalogResponse(context: RestLibraryCatalogRouteContext) throws -> RestHttpResponse {
        let projections: Array<LibraryCatalogEntryProjection> = LibraryCatalogProjection.projectCatalogEntries(
            downloadCatalog: context.downloadCatalog,
            discoveredModels: context.discoveredModelsProvider(),
            validatedPublications: context.validatedPublicationsProvider(),
            currentJob: context.currentJobProvider())
        var entryValues: Array<JsonWireValue> = Array()
        entryValues.reserveCapacity(projections.count)
        for projection: LibraryCatalogEntryProjection in projections {
            let destinationDirectory: String? = projection.discoveredModelDirectory
                ?? context.destinationDirectoryProvider(projection.catalogEntry.huggingfaceId)
            entryValues.append(LibraryCatalogEndpoint.entryValue(
                projection: projection,
                destinationDirectory: destinationDirectory))
        }
        var responseObject: JsonWireObject = JsonWireObject(entries: Array())
        responseObject.appendEntry(
            key: "schema_version",
            value: .unsignedInteger(UInt64(context.downloadCatalog.schemaVersion)))
        responseObject.appendEntry(key: "entries", value: .array(entryValues))
        return try RestHttpResponse.json(statusCode: 200, wireValue: .object(responseObject))
    }

    private static func entryValue(
        projection: LibraryCatalogEntryProjection,
        destinationDirectory: String?
    ) -> JsonWireValue {
        let catalogEntry: DownloadCatalogEntry = projection.catalogEntry
        var entryObject: JsonWireObject = JsonWireObject(entries: Array())
        entryObject.appendEntry(key: "huggingface_id", value: .string(catalogEntry.huggingfaceId))
        entryObject.appendEntry(key: "revision", value: .string(catalogEntry.revision))
        entryObject.appendEntry(key: "display_name", value: .string(catalogEntry.displayName))
        entryObject.appendEntry(key: "family", value: .string(catalogEntry.family.rawValue))
        entryObject.appendEntry(
            key: "approximate_size_bytes",
            value: .unsignedInteger(catalogEntry.approximateSizeBytes))
        entryObject.appendEntry(key: "public", value: .boolean(true))
        entryObject.appendEntry(key: "ready_on_this_mac", value: .boolean(projection.readyOnThisMac))
        if let destinationDirectory: String = destinationDirectory {
            entryObject.appendEntry(key: "destination_directory", value: .string(destinationDirectory))
        }
        entryObject.appendEntry(
            key: "download_state",
            value: projection.downloadState.map({ (stateName: String) -> JsonWireValue in
                return .string(stateName)
            }) ?? .null)
        if let descriptionText: String = catalogEntry.description {
            entryObject.appendEntry(key: "description", value: .string(descriptionText))
        }
        if let quantizationLabel: String = catalogEntry.quantizationLabel {
            entryObject.appendEntry(key: "quantization_label", value: .string(quantizationLabel))
        }
        if let architectureSummary: String = catalogEntry.architectureSummary {
            entryObject.appendEntry(key: "architecture_summary", value: .string(architectureSummary))
        }
        if let upstreamLicense: String = catalogEntry.upstreamLicense {
            entryObject.appendEntry(key: "upstream_license", value: .string(upstreamLicense))
        }
        if let requestableModelId: String = projection.requestableModelId {
            entryObject.appendEntry(key: "requestable_model_id", value: .string(requestableModelId))
        }
        entryObject.appendEntry(
            key: "capabilities",
            value: LibraryCatalogEndpoint.capabilitiesValue(projection.capabilities))
        return .object(entryObject)
    }

    private static func capabilitiesValue(_ capabilities: DownloadCatalogCapabilities) -> JsonWireValue {
        var capabilitiesObject: JsonWireObject = JsonWireObject(entries: Array())
        capabilitiesObject.appendEntry(key: "supports_reasoning", value: .boolean(capabilities.supportsReasoning))
        capabilitiesObject.appendEntry(key: "supports_vision", value: .boolean(capabilities.supportsVision))
        capabilitiesObject.appendEntry(key: "supports_tool_calls", value: .boolean(capabilities.supportsToolCalls))
        capabilitiesObject.appendEntry(
            key: "supports_image_generation",
            value: .boolean(capabilities.supportsImageGeneration))
        capabilitiesObject.appendEntry(key: "supports_embeddings", value: .boolean(capabilities.supportsEmbeddings))
        if let contextWindow: UInt32 = capabilities.contextWindow {
            capabilitiesObject.appendEntry(key: "context_window", value: .unsignedInteger(UInt64(contextWindow)))
        }
        if let maxOutputTokens: UInt32 = capabilities.maxOutputTokens {
            capabilitiesObject.appendEntry(key: "max_output_tokens", value: .unsignedInteger(UInt64(maxOutputTokens)))
        }
        return .object(capabilitiesObject)
    }
}
