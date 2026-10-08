import Foundation

import MLX
import RuntimeIntegration

/**
 * Loads only the validated tensor ranges described by one expert-page
 * manifest, port of the Rust `load_quantized_expert_page`.
 *
 * A model shard can contain far more expert data than one forward needs,
 * so asking the ordinary SafeTensors loader to open the whole shard would
 * make a routed top-K decode miss behave like a whole-shard load. The
 * validated manifest instead describes exactly the source byte intervals
 * for the selected experts; this loader turns that description into named
 * MLX arrays without teaching the bounded reader anything about Qwen
 * projections.
 *
 * One logical page may span several shards, so tensors are merged only
 * after each shard has independently passed bounded-header construction.
 * A tensor name published by more than one shard is an invalid manifest:
 * silently replacing an array could pair one projection's weight with
 * another projection's metadata.
 *
 * Documented divergence from Rust: runtime failures (`MlxRuntimeError`)
 * propagate unchanged instead of being re-wrapped in a lossy string,
 * which Swift's typed errors make unnecessary.
 */
public enum QuantizedExpertPageLoader {

    /**
     * Loads the named tensors of one validated expert page.
     *
     * - Parameters:
     *   - modelDirectory: directory holding the shard files the page's
     *     manifests name.
     *   - pageManifest: the validated page assembly plan.
     *   - expertFileReadMetrics: optional instrumentation; every bounded
     *     read is measured for overlap, latency, and volume when
     *     attached. Measured counts are not proof of equal physical SSD
     *     traffic: macOS may satisfy reads from its file cache.
     * - Returns: every tensor the page publishes, keyed by safetensors
     *   name. Arrays stay addressable through the loader's retained
     *   buffers; the architecture-specific page builder gives them their
     *   gate/up/down meaning.
     * - Throws: `ExpertPagingError.manifestValidationFailure` when a
     *   shard file cannot be opened or a tensor name repeats across
     *   shards; `MlxRuntimeError` propagates unchanged from the bounded
     *   load or a tensor lookup.
     */
    public static func loadPage(
        modelDirectory: URL,
        pageManifest: QuantizedExpertPageManifest,
        expertFileReadMetrics: PositionalFileReadMetrics?
    ) throws -> [String: MLXArray] {
        var loadedTensorsByName: [String: MLXArray] = [:]
        for shardManifest: QuantizedExpertShardManifest in pageManifest.sourceManifests {
            let shardTensorsByName: [String: MLXArray] = try QuantizedExpertPageLoader.loadShardManifest(
                modelDirectory: modelDirectory,
                shardManifest: shardManifest,
                expertFileReadMetrics: expertFileReadMetrics)
            for (tensorName, tensor): (String, MLXArray) in shardTensorsByName {
                if loadedTensorsByName.updateValue(tensor, forKey: tensorName) != nil {
                    throw ExpertPagingError.manifestValidationFailure(
                        description: "the tensor \(tensorName) is published by more than one shard of the page")
                }
            }
        }
        return loadedTensorsByName
    }

    private static func loadShardManifest(
        modelDirectory: URL,
        shardManifest: QuantizedExpertShardManifest,
        expertFileReadMetrics: PositionalFileReadMetrics?
    ) throws -> [String: MLXArray] {
        // The synthetic header names only the selected tensors and rebases
        // them into a dense virtual address space; the source intervals are
        // the translation table from that virtual space back to disjoint
        // offsets inside the real shard file.
        let syntheticHeaderBytes: Data = try shardManifest.rebasedSafetensorsHeader()
        let shardFileUrl: URL = modelDirectory.appendingPathComponent(shardManifest.sourceFileName)
        let sourceFile: FileHandle
        do {
            sourceFile = try FileHandle(forReadingFrom: shardFileUrl)
        } catch {
            throw ExpertPagingError.manifestValidationFailure(
                description: "the shard file \(shardManifest.sourceFileName) could not be opened: \(error)")
        }
        defer {
            try? sourceFile.close()
        }
        let boundedIntervals: [BoundedReadInterval] = shardManifest.sourceIntervals.map({ (sourceInterval: QuantizedExpertSourceInterval) -> BoundedReadInterval in
            return BoundedReadInterval(
                virtualPayloadOffset: sourceInterval.virtualPayloadOffsetBytes,
                sourceFileOffset: sourceInterval.sourceFileOffsetBytes,
                sourceByteCount: sourceInterval.sourceByteCount)
        })
        let loadResult: SafetensorsFile = try MlxRuntime.loadSafetensorsFromBoundedRanges(
            sourceFile: sourceFile,
            syntheticHeaderBytes: syntheticHeaderBytes,
            intervals: boundedIntervals,
            totalPayloadBytes: shardManifest.payloadByteCount,
            expertFileReadMetrics: expertFileReadMetrics)
        // Iterate the validated plan — not every tensor the synthetic
        // loader happens to expose — so the architecture layer receives
        // exactly the names it requested and a missing lazy array becomes
        // a typed failure at this boundary.
        var shardTensorsByName: [String: MLXArray] = Dictionary(
            minimumCapacity: shardManifest.tensorRanges.count)
        for tensorRange: QuantizedExpertTensorRange in shardManifest.tensorRanges {
            shardTensorsByName[tensorRange.tensorName] = try loadResult.tensor(tensorRange.tensorName)
        }
        return shardTensorsByName
    }
}
