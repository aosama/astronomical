import Foundation

import Testing

import MLX
import JourneyCategories
import ModelServing
import ModelServingTestSupport
import RuntimeIntegration

/// Hermetic journeys over the bounded expert page loader, port of the
/// coverage the Rust real-model parity journeys apply to
/// `load_quantized_expert_page`: the loader reads exactly the validated
/// tensor runs a page manifest describes, merges shards without name
/// collisions, attaches positional-read metrics, and fails closed on an
/// invalid manifest instead of pairing one projection's weight with
/// another projection's metadata.
@Suite(.serialized, .tags(.hermeticMlxJourney))
final class QuantizedExpertPageLoaderTests {

    init() {
        MLXMetallibLocator.overrideMetallibPathIfNecessary()
    }

    private struct ShardTensorSpec {
        let tensorName: String
        let projectionName: String
        let parameterName: String
        let shape: [Int]
        let fillBaseByte: UInt8

        var byteCount: Int {
            return self.shape.reduce(1, *)
        }
    }

    private static let SHARD_A_FILE_NAME: String = "fictional-shard-a.safetensors"
    private static let SHARD_B_FILE_NAME: String = "fictional-shard-b.safetensors"

    @Test
    func should_load_the_named_tensors_of_a_single_shard_page() throws {
        let shardTensors: [ShardTensorSpec] = [
            ShardTensorSpec(
                tensorName: "fictional.layers.0.switch_mlp.gate.weight",
                projectionName: "gate", parameterName: "weight",
                shape: [4], fillBaseByte: 0),
            ShardTensorSpec(
                tensorName: "fictional.layers.0.switch_mlp.up.weight",
                projectionName: "up", parameterName: "weight",
                shape: [3], fillBaseByte: 10),
        ]
        let pageFixture: PageLoaderFixture = try PageLoaderFixture.build(
            shardFileName: QuantizedExpertPageLoaderTests.SHARD_A_FILE_NAME,
            shardTensors: shardTensors)
        defer { pageFixture.cleanUp() }
        let readMetrics: PositionalFileReadMetrics = PositionalFileReadMetrics()

        let loadedTensorsByName: [String: MLXArray] = try QuantizedExpertPageLoader.loadPage(
            modelDirectory: pageFixture.modelDirectoryUrl,
            pageManifest: pageFixture.pageManifest,
            expertFileReadMetrics: readMetrics)

        #expect(Set(loadedTensorsByName.keys) == Set([
            "fictional.layers.0.switch_mlp.gate.weight",
            "fictional.layers.0.switch_mlp.up.weight",
        ]))
        try QuantizedExpertPageLoaderTests.assertTensorValues(
            loadedTensorsByName: loadedTensorsByName,
            tensorName: "fictional.layers.0.switch_mlp.gate.weight",
            expectedFillBaseByte: 0, expectedElementCount: 4)
        try QuantizedExpertPageLoaderTests.assertTensorValues(
            loadedTensorsByName: loadedTensorsByName,
            tensorName: "fictional.layers.0.switch_mlp.up.weight",
            expectedFillBaseByte: 10, expectedElementCount: 3)
        let metricsSnapshot: PositionalFileReadSnapshot = readMetrics.snapshot()
        #expect(metricsSnapshot.readByteCount == 7)
        #expect(metricsSnapshot.readFailureCount == 0)
    }

    @Test
    func should_assemble_a_page_spanning_two_shards() throws {
        let shardATensors: [ShardTensorSpec] = [
            ShardTensorSpec(
                tensorName: "fictional.layers.0.switch_mlp.gate.weight",
                projectionName: "gate", parameterName: "weight",
                shape: [2], fillBaseByte: 0),
        ]
        let shardBTensors: [ShardTensorSpec] = [
            ShardTensorSpec(
                tensorName: "fictional.layers.0.switch_mlp.down.weight",
                projectionName: "down", parameterName: "weight",
                shape: [2], fillBaseByte: 40),
        ]
        let pageFixture: PageLoaderFixture = try PageLoaderFixture.build(
            shardFileName: QuantizedExpertPageLoaderTests.SHARD_A_FILE_NAME,
            shardTensors: shardATensors,
            secondShardFileName: QuantizedExpertPageLoaderTests.SHARD_B_FILE_NAME,
            secondShardTensors: shardBTensors)
        defer { pageFixture.cleanUp() }

        let loadedTensorsByName: [String: MLXArray] = try QuantizedExpertPageLoader.loadPage(
            modelDirectory: pageFixture.modelDirectoryUrl,
            pageManifest: pageFixture.pageManifest,
            expertFileReadMetrics: nil)

        try QuantizedExpertPageLoaderTests.assertTensorValues(
            loadedTensorsByName: loadedTensorsByName,
            tensorName: "fictional.layers.0.switch_mlp.gate.weight",
            expectedFillBaseByte: 0, expectedElementCount: 2)
        try QuantizedExpertPageLoaderTests.assertTensorValues(
            loadedTensorsByName: loadedTensorsByName,
            tensorName: "fictional.layers.0.switch_mlp.down.weight",
            expectedFillBaseByte: 40, expectedElementCount: 2)
    }

    @Test
    func should_read_only_the_selected_expert_run_within_one_tensor() throws {
        let shardTensors: [ShardTensorSpec] = [
            ShardTensorSpec(
                tensorName: "fictional.layers.0.switch_mlp.gate.weight",
                projectionName: "gate", parameterName: "weight",
                shape: [4], fillBaseByte: 0),
        ]
        let pageFixture: PageLoaderFixture = try PageLoaderFixture.build(
            shardFileName: QuantizedExpertPageLoaderTests.SHARD_A_FILE_NAME,
            shardTensors: shardTensors)
        defer { pageFixture.cleanUp() }
        let readMetrics: PositionalFileReadMetrics = PositionalFileReadMetrics()
        // Select experts 1..<3 inside the leading (expert) axis: the page
        // carries only bytes [1, 3) of the four-element tensor.
        let slicedManifest: QuantizedExpertPageManifest = QuantizedExpertPageLoaderTests.slicedRunPageManifest(
            shardFileName: QuantizedExpertPageLoaderTests.SHARD_A_FILE_NAME,
            payloadStartByte: pageFixture.payloadStartByte,
            tensorSpec: shardTensors[0],
            expertStart: 1, expertCount: 2)

        let loadedTensorsByName: [String: MLXArray] = try QuantizedExpertPageLoader.loadPage(
            modelDirectory: pageFixture.modelDirectoryUrl,
            pageManifest: slicedManifest,
            expertFileReadMetrics: readMetrics)

        try QuantizedExpertPageLoaderTests.assertTensorValues(
            loadedTensorsByName: loadedTensorsByName,
            tensorName: "fictional.layers.0.switch_mlp.gate.weight",
            expectedFillBaseByte: 1, expectedElementCount: 2)
        let metricsSnapshot: PositionalFileReadSnapshot = readMetrics.snapshot()
        #expect(metricsSnapshot.readByteCount == 2)
        #expect(metricsSnapshot.readFailureCount == 0)
    }

    @Test
    func should_fail_closed_when_a_tensor_name_is_duplicated_across_shards() throws {
        let duplicatedTensorName: String = "fictional.layers.0.switch_mlp.gate.weight"
        let shardATensors: [ShardTensorSpec] = [
            ShardTensorSpec(
                tensorName: duplicatedTensorName,
                projectionName: "gate", parameterName: "weight",
                shape: [2], fillBaseByte: 0),
        ]
        let shardBTensors: [ShardTensorSpec] = [
            ShardTensorSpec(
                tensorName: duplicatedTensorName,
                projectionName: "gate", parameterName: "weight",
                shape: [2], fillBaseByte: 50),
        ]
        let pageFixture: PageLoaderFixture = try PageLoaderFixture.build(
            shardFileName: QuantizedExpertPageLoaderTests.SHARD_A_FILE_NAME,
            shardTensors: shardATensors,
            secondShardFileName: QuantizedExpertPageLoaderTests.SHARD_B_FILE_NAME,
            secondShardTensors: shardBTensors)
        defer { pageFixture.cleanUp() }

        do {
            _ = try QuantizedExpertPageLoader.loadPage(
                modelDirectory: pageFixture.modelDirectoryUrl,
                pageManifest: pageFixture.pageManifest,
                expertFileReadMetrics: nil)
            Issue.record("a tensor name published by two shards must fail closed")
        } catch let pagingError as ExpertPagingError {
            guard case .manifestValidationFailure(let failureDescription) = pagingError else {
                Issue.record("expected a manifest validation failure, found \(pagingError)")
                return
            }
            #expect(failureDescription.contains("more than one shard"))
        }
    }

    @Test
    func should_propagate_bounded_interval_failures_unchanged() throws {
        let shardTensors: [ShardTensorSpec] = [
            ShardTensorSpec(
                tensorName: "fictional.layers.0.switch_mlp.gate.weight",
                projectionName: "gate", parameterName: "weight",
                shape: [4], fillBaseByte: 0),
        ]
        let pageFixture: PageLoaderFixture = try PageLoaderFixture.build(
            shardFileName: QuantizedExpertPageLoaderTests.SHARD_A_FILE_NAME,
            shardTensors: shardTensors)
        defer { pageFixture.cleanUp() }
        // The shard plan promises a four-byte virtual payload but the
        // interval chain supplies only two, so the bounded reader's tiling
        // validation must reject the load before any bytes are read.
        let inconsistentManifest: QuantizedExpertPageManifest = QuantizedExpertPageManifest(
            expertIds: [0],
            pageSlotByGlobalExpertId: [0],
            sourceManifests: [
                QuantizedExpertShardManifest(
                    sourceFileName: QuantizedExpertPageLoaderTests.SHARD_A_FILE_NAME,
                    tensorRanges: [
                        QuantizedExpertTensorRange(
                            tensorName: shardTensors[0].tensorName,
                            projectionName: shardTensors[0].projectionName,
                            parameterName: shardTensors[0].parameterName,
                            dtype: .u8, shape: [4],
                            virtualPayloadOffsetBytes: 0, byteCount: 4),
                    ],
                    sourceIntervals: [
                        QuantizedExpertSourceInterval(
                            tensorName: shardTensors[0].tensorName,
                            expertStart: 0, expertCount: 2,
                            sourceFileOffsetBytes: UInt64(pageFixture.payloadStartByte),
                            sourceByteCount: 2,
                            virtualPayloadOffsetBytes: 0),
                    ],
                    payloadByteCount: 4),
            ],
            payloadByteCount: 4)

        do {
            _ = try QuantizedExpertPageLoader.loadPage(
                modelDirectory: pageFixture.modelDirectoryUrl,
                pageManifest: inconsistentManifest,
                expertFileReadMetrics: nil)
            Issue.record("an untiled virtual payload must fail closed")
        } catch let runtimeError as MlxRuntimeError {
            guard case .boundedIntervalValidation = runtimeError else {
                Issue.record("expected a bounded interval validation failure, found \(runtimeError)")
                return
            }
        }
    }

    @Test
    func should_fail_closed_when_a_shard_file_is_missing() throws {
        let shardTensors: [ShardTensorSpec] = [
            ShardTensorSpec(
                tensorName: "fictional.layers.0.switch_mlp.gate.weight",
                projectionName: "gate", parameterName: "weight",
                shape: [2], fillBaseByte: 0),
        ]
        let pageFixture: PageLoaderFixture = try PageLoaderFixture.build(
            shardFileName: QuantizedExpertPageLoaderTests.SHARD_A_FILE_NAME,
            shardTensors: shardTensors)
        defer { pageFixture.cleanUp() }
        try FileManager.default.removeItem(
            at: pageFixture.modelDirectoryUrl
                .appendingPathComponent(QuantizedExpertPageLoaderTests.SHARD_A_FILE_NAME))

        do {
            _ = try QuantizedExpertPageLoader.loadPage(
                modelDirectory: pageFixture.modelDirectoryUrl,
                pageManifest: pageFixture.pageManifest,
                expertFileReadMetrics: nil)
            Issue.record("a page naming a missing shard file must fail closed")
        } catch let pagingError as ExpertPagingError {
            guard case .manifestValidationFailure(let failureDescription) = pagingError else {
                Issue.record("expected a manifest validation failure, found \(pagingError)")
                return
            }
            #expect(failureDescription.contains("could not be opened"))
        }
    }

    /// Writes real shard files and the full-run page manifest over them.
    private final class PageLoaderFixture {

        let modelDirectoryUrl: URL
        let pageManifest: QuantizedExpertPageManifest
        let payloadStartByte: Int

        private init(modelDirectoryUrl: URL, pageManifest: QuantizedExpertPageManifest, payloadStartByte: Int) {
            self.modelDirectoryUrl = modelDirectoryUrl
            self.pageManifest = pageManifest
            self.payloadStartByte = payloadStartByte
        }

        fileprivate static func build(
            shardFileName: String,
            shardTensors: [ShardTensorSpec],
            secondShardFileName: String? = nil,
            secondShardTensors: [ShardTensorSpec]? = nil
        ) throws -> PageLoaderFixture {
            let modelDirectoryUrl: URL = FileManager.default.temporaryDirectory
                .appendingPathComponent("page-loader-\(UUID().uuidString)")
            try FileManager.default.createDirectory(at: modelDirectoryUrl, withIntermediateDirectories: true)
            var sourceManifests: [QuantizedExpertShardManifest] = []
            let firstShardBuild: ShardBuild = try QuantizedExpertPageLoaderTests.fullRunSourceManifest(
                modelDirectoryUrl: modelDirectoryUrl,
                shardFileName: shardFileName,
                shardTensors: shardTensors)
            sourceManifests.append(firstShardBuild.sourceManifest)
            if let secondShardFileName: String = secondShardFileName,
                let secondShardTensors: [ShardTensorSpec] = secondShardTensors {
                let secondShardBuild: ShardBuild = try QuantizedExpertPageLoaderTests.fullRunSourceManifest(
                    modelDirectoryUrl: modelDirectoryUrl,
                    shardFileName: secondShardFileName,
                    shardTensors: secondShardTensors)
                sourceManifests.append(secondShardBuild.sourceManifest)
            }
            let payloadByteCount: UInt64 = sourceManifests.reduce(0, { (sumBytes: UInt64, sourceManifest: QuantizedExpertShardManifest) -> UInt64 in
                return sumBytes + sourceManifest.payloadByteCount
            })
            let pageManifest: QuantizedExpertPageManifest = QuantizedExpertPageManifest(
                expertIds: [0],
                pageSlotByGlobalExpertId: [0],
                sourceManifests: sourceManifests,
                payloadByteCount: payloadByteCount)
            return PageLoaderFixture(
                modelDirectoryUrl: modelDirectoryUrl,
                pageManifest: pageManifest,
                payloadStartByte: firstShardBuild.payloadStartByte)
        }

        fileprivate func cleanUp() -> Void {
            try? FileManager.default.removeItem(at: self.modelDirectoryUrl)
        }
    }

    /// One written shard file plus the manifest over it, with the
    /// absolute payload offset the file's header ends at.
    private struct ShardBuild {
        let sourceManifest: QuantizedExpertShardManifest
        let payloadStartByte: Int
    }

    private static func fullRunSourceManifest(
        modelDirectoryUrl: URL,
        shardFileName: String,
        shardTensors: [ShardTensorSpec]
    ) throws -> ShardBuild {
        let payloadStartByte: Int = try QuantizedExpertPageLoaderTests.writeShardFile(
            fileUrl: modelDirectoryUrl.appendingPathComponent(shardFileName),
            tensors: shardTensors)
        var tensorRanges: [QuantizedExpertTensorRange] = []
        var sourceIntervals: [QuantizedExpertSourceInterval] = []
        var tensorStartOffsetBytes: Int = 0
        for tensorSpec: ShardTensorSpec in shardTensors {
            tensorRanges.append(QuantizedExpertTensorRange(
                tensorName: tensorSpec.tensorName,
                projectionName: tensorSpec.projectionName,
                parameterName: tensorSpec.parameterName,
                dtype: .u8,
                shape: tensorSpec.shape,
                virtualPayloadOffsetBytes: UInt64(tensorStartOffsetBytes),
                byteCount: tensorSpec.byteCount))
            sourceIntervals.append(QuantizedExpertSourceInterval(
                tensorName: tensorSpec.tensorName,
                expertStart: 0,
                expertCount: tensorSpec.shape[0],
                sourceFileOffsetBytes: UInt64(payloadStartByte + tensorStartOffsetBytes),
                sourceByteCount: tensorSpec.byteCount,
                virtualPayloadOffsetBytes: UInt64(tensorStartOffsetBytes)))
            tensorStartOffsetBytes += tensorSpec.byteCount
        }
        return ShardBuild(
            sourceManifest: QuantizedExpertShardManifest(
                sourceFileName: shardFileName,
                tensorRanges: tensorRanges,
                sourceIntervals: sourceIntervals,
                payloadByteCount: UInt64(tensorStartOffsetBytes)),
            payloadStartByte: payloadStartByte)
    }

    private static func slicedRunPageManifest(
        shardFileName: String,
        payloadStartByte: Int,
        tensorSpec: ShardTensorSpec,
        expertStart: Int,
        expertCount: Int
    ) -> QuantizedExpertPageManifest {
        let slicedShape: [Int] = [expertCount]
        let slicedByteCount: Int = expertCount
        return QuantizedExpertPageManifest(
            expertIds: [expertStart],
            pageSlotByGlobalExpertId: [
                QuantizedExpertPageManifest.ABSENT_PAGE_SLOT, 0,
                QuantizedExpertPageManifest.ABSENT_PAGE_SLOT, QuantizedExpertPageManifest.ABSENT_PAGE_SLOT,
            ],
            sourceManifests: [
                QuantizedExpertShardManifest(
                    sourceFileName: shardFileName,
                    tensorRanges: [
                        QuantizedExpertTensorRange(
                            tensorName: tensorSpec.tensorName,
                            projectionName: tensorSpec.projectionName,
                            parameterName: tensorSpec.parameterName,
                            dtype: .u8,
                            shape: slicedShape,
                            virtualPayloadOffsetBytes: 0,
                            byteCount: slicedByteCount),
                    ],
                    sourceIntervals: [
                        QuantizedExpertSourceInterval(
                            tensorName: tensorSpec.tensorName,
                            expertStart: expertStart,
                            expertCount: expertCount,
                            sourceFileOffsetBytes: UInt64(payloadStartByte + expertStart),
                            sourceByteCount: slicedByteCount,
                            virtualPayloadOffsetBytes: 0),
                    ],
                    payloadByteCount: UInt64(slicedByteCount)),
            ],
            payloadByteCount: UInt64(slicedByteCount))
    }

    private static func writeShardFile(fileUrl: URL, tensors: [ShardTensorSpec]) throws -> Int {
        var payloadBytes: Data = Data()
        var headerEntries: [String] = []
        for tensorSpec: ShardTensorSpec in tensors {
            let payloadStartByte: Int = payloadBytes.count
            for elementPosition: Int in 0 ..< tensorSpec.byteCount {
                payloadBytes.append(tensorSpec.fillBaseByte &+ UInt8(truncatingIfNeeded: elementPosition))
            }
            let payloadEndByte: Int = payloadBytes.count
            let shapeText: String = tensorSpec.shape
                .map({ (dimension: Int) -> String in String(dimension) })
                .joined(separator: ",")
            headerEntries.append(
                "\"\(tensorSpec.tensorName)\": {\"dtype\": \"U8\", "
                    + "\"shape\": [\(shapeText)], "
                    + "\"data_offsets\": [\(payloadStartByte), \(payloadEndByte)]}")
        }
        let headerJsonText: String = "{" + headerEntries.joined(separator: ",") + "}"
        let headerBytes: Data = Data(headerJsonText.utf8)
        var fileBytes: Data = Data()
        withUnsafeBytes(of: UInt64(headerBytes.count).littleEndian, { (lengthBuffer: UnsafeRawBufferPointer) -> Void in
            fileBytes.append(contentsOf: lengthBuffer)
        })
        fileBytes.append(headerBytes)
        fileBytes.append(payloadBytes)
        try fileBytes.write(to: fileUrl)
        return 8 + headerBytes.count
    }

    private static func assertTensorValues(
        loadedTensorsByName: [String: MLXArray],
        tensorName: String,
        expectedFillBaseByte: UInt8,
        expectedElementCount: Int
    ) throws -> Void {
        let loadedTensor: MLXArray = try #require(loadedTensorsByName[tensorName])
        let loadedValues: [UInt8] = loadedTensor.asArray(UInt8.self)
        let expectedValues: [UInt8] = (0 ..< expectedElementCount).map({ (elementPosition: Int) -> UInt8 in
            return expectedFillBaseByte &+ UInt8(truncatingIfNeeded: elementPosition)
        })
        #expect(loadedValues == expectedValues)
    }
}
