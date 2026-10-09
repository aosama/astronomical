import Foundation

import Testing

import MLX
import JourneyCategories
import ModelServing
import ModelServingTestSupport
import RuntimeIntegration

/// Hermetic journeys over the disk-backed expert page materializer: the
/// shard-ranges-only residency seam the paged decorator consults. Every
/// journey reads a real temporary SafeTensors shard through the bounded
/// loader and asserts per-expert slice identity, projection parameter
/// coverage (affine triplets versus native weight-only), read-volume
/// accounting, and fail-closed behavior on invalid layer requests.
extension MlxGpuJourneyContainer {

    @Suite(.tags(.hermeticMlxJourney))
    final class Qwen35MoeDiskExpertPageMaterializerTests {

    init() {
        MLXMetallibLocator.overrideMetallibPathIfNecessary()
    }

    private static let SHARD_FILE_NAME: String = "fictional-experts.safetensors"
    private static let LAYER_PREFIX: String = "fictional.layer.0"
    private static let EXPERT_CAPACITY: Int = 4
    private static let SLICE_ROWS: Int = 3
    private static let SLICE_COLUMNS: Int = 2
    private static let EXPERT_PAYLOAD_BYTES: Int = SLICE_ROWS * SLICE_COLUMNS
    private static let SLICE_SHAPE: [Int] = [SLICE_ROWS, SLICE_COLUMNS]

    @Test
    func should_materialize_affine_slices_for_two_sparse_experts() throws {
        let fixture: MaterializerFixture = try MaterializerFixture.build(
            projectionStorageByName: [
                "gate_proj": .affine, "up_proj": .affine, "down_proj": .affine,
            ],
            expertIds: [0, 2])
        defer { fixture.cleanUp() }

        let materializedWeights: Qwen35MoeMaterializedExpertWeights = try fixture
            .materializer.materializeExpertWeights(layerIndex: 0, expertIds: [0, 2])

        #expect(materializedWeights.expertPageReadCount == 1)
        try Qwen35MoeDiskExpertPageMaterializerTests.assertProjectionSlices(
            materializedWeights.gateProjection,
            projectionName: "gate_proj", fillBaseByte: 0,
            expertIds: [0, 2])
        try Qwen35MoeDiskExpertPageMaterializerTests.assertProjectionSlices(
            materializedWeights.upProjection,
            projectionName: "up_proj", fillBaseByte: 40,
            expertIds: [0, 2])
        try Qwen35MoeDiskExpertPageMaterializerTests.assertProjectionSlices(
            materializedWeights.downProjection,
            projectionName: "down_proj", fillBaseByte: 80,
            expertIds: [0, 2])
    }

    @Test
    func should_materialize_weight_only_slices_for_native_projections() throws {
        let fixture: MaterializerFixture = try MaterializerFixture.build(
            projectionStorageByName: [
                "gate_proj": .nativeBfloat16,
                "up_proj": .nativeBfloat16,
                "down_proj": .nativeBfloat16,
            ],
            expertIds: [3])
        defer { fixture.cleanUp() }

        let materializedWeights: Qwen35MoeMaterializedExpertWeights = try fixture
            .materializer.materializeExpertWeights(layerIndex: 0, expertIds: [3])

        #expect(materializedWeights.expertPageReadCount == 1)
        for (projectionSlices, projectionName, fillBaseByte): (
            Qwen35MoeMaterializedProjectionSlices, String, UInt8
        ) in [
            (materializedWeights.gateProjection, "gate_proj", UInt8(0)),
            (materializedWeights.upProjection, "up_proj", UInt8(40)),
            (materializedWeights.downProjection, "down_proj", UInt8(80)),
        ] {
            try Qwen35MoeDiskExpertPageMaterializerTests.assertProjectionSlices(
                projectionSlices,
                projectionName: projectionName, fillBaseByte: fillBaseByte,
                expertIds: [3],
                expectedParameterBasenames: ["weight"])
        }
    }

    @Test
    func should_materialize_mixed_projections_in_one_layer() throws {
        let fixture: MaterializerFixture = try MaterializerFixture.build(
            projectionStorageByName: [
                "gate_proj": .nativeBfloat16, "up_proj": .affine, "down_proj": .affine,
            ],
            expertIds: [1])
        defer { fixture.cleanUp() }

        let materializedWeights: Qwen35MoeMaterializedExpertWeights = try fixture
            .materializer.materializeExpertWeights(layerIndex: 0, expertIds: [1])

        #expect(materializedWeights.expertPageReadCount == 1)
        try Qwen35MoeDiskExpertPageMaterializerTests.assertProjectionSlices(
            materializedWeights.gateProjection,
            projectionName: "gate_proj", fillBaseByte: 0,
            expertIds: [1],
            expectedParameterBasenames: ["weight"])
        try Qwen35MoeDiskExpertPageMaterializerTests.assertProjectionSlices(
            materializedWeights.upProjection,
            projectionName: "up_proj", fillBaseByte: 40,
            expertIds: [1])
        try Qwen35MoeDiskExpertPageMaterializerTests.assertProjectionSlices(
            materializedWeights.downProjection,
            projectionName: "down_proj", fillBaseByte: 80,
            expertIds: [1])
    }

    @Test
    func should_account_read_bytes_against_the_page_payload() throws {
        let readMetrics: PositionalFileReadMetrics = PositionalFileReadMetrics()
        let fixture: MaterializerFixture = try MaterializerFixture.build(
            projectionStorageByName: [
                "gate_proj": .affine, "up_proj": .affine, "down_proj": .affine,
            ],
            expertIds: [1, 3],
            expertFileReadMetrics: readMetrics)
        defer { fixture.cleanUp() }

        let materializedWeights: Qwen35MoeMaterializedExpertWeights = try fixture
            .materializer.materializeExpertWeights(layerIndex: 0, expertIds: [1, 3])

        // Every selected expert row of every layer tensor was read, and
        // nothing outside the page manifest's payload: the count must
        // equal the manifest's own byte accounting.
        #expect(readMetrics.snapshot().readByteCount == fixture.pagePayloadByteCount)
        #expect(materializedWeights.expertPageReadCount == 1)
        try Qwen35MoeDiskExpertPageMaterializerTests.assertProjectionSlices(
            materializedWeights.gateProjection,
            projectionName: "gate_proj", fillBaseByte: 0,
            expertIds: [1, 3])
    }

    @Test
    func should_fail_closed_when_the_layer_index_is_out_of_range() throws {
        let fixture: MaterializerFixture = try MaterializerFixture.build(
            projectionStorageByName: [
                "gate_proj": .affine, "up_proj": .affine, "down_proj": .affine,
            ],
            expertIds: [0])
        defer { fixture.cleanUp() }

        do {
            _ = try fixture.materializer.materializeExpertWeights(layerIndex: 1, expertIds: [0])
            Issue.record("a layer index beyond the startup plans must fail closed")
        } catch let pagingError as ExpertPagingError {
            #expect(pagingError == .layerIndexOutOfRange(layerIndex: 1, layerCount: 1))
        }
    }

    @Test
    func should_fail_closed_when_an_affine_page_tensor_is_missing() throws {
        // A hand-built plan that declares affine storage but carries only
        // weight sources: the materializer must reject the incomplete page
        // instead of synthesizing scales or biases.
        let fixture: MaterializerFixture = try MaterializerFixture.build(
            projectionStorageByName: [
                "gate_proj": .affine, "up_proj": .affine, "down_proj": .affine,
            ],
            expertIds: [0],
            writeAffineCompanionTensors: false)
        defer { fixture.cleanUp() }

        do {
            _ = try fixture.materializer.materializeExpertWeights(layerIndex: 0, expertIds: [0])
            Issue.record("an affine page missing a companion tensor must fail closed")
        } catch let pagingError as ExpertPagingError {
            #expect(pagingError == .pageTensorMissing(
                tensorName: "gate_proj.scales",
                layerPrefix: Qwen35MoeDiskExpertPageMaterializerTests.LAYER_PREFIX))
        }
    }

    private static func assertProjectionSlices(
        _ projectionSlices: Qwen35MoeMaterializedProjectionSlices,
        projectionName: String,
        fillBaseByte: UInt8,
        expertIds: [Int],
        expectedParameterBasenames: [String] = ["weight", "scales", "biases"]
    ) throws -> Void {
        #expect(
            Set(projectionSlices.parametersByParameterBasename.keys)
                == Set(expectedParameterBasenames))
        for parameterBasename: String in expectedParameterBasenames {
            let expertSlicesByExpertId: [Int: MLXArray] = try #require(
                projectionSlices.parametersByParameterBasename[parameterBasename])
            #expect(Set(expertSlicesByExpertId.keys) == Set(expertIds))
            for expertId: Int in expertIds {
                let expertSlice: MLXArray = try #require(expertSlicesByExpertId[expertId])
                #expect(expertSlice.shape == Qwen35MoeDiskExpertPageMaterializerTests.SLICE_SHAPE)
                let expectedRowByte: UInt8 = fillBaseByte &+ UInt8(truncatingIfNeeded: expertId)
                let expectedValues: [UInt8] = [UInt8](
                    repeating: expectedRowByte,
                    count: Qwen35MoeDiskExpertPageMaterializerTests.EXPERT_PAYLOAD_BYTES)
                #expect(expertSlice.asArray(UInt8.self) == expectedValues)
            }
        }
    }

    /// Writes one real shard file and hands back a materializer over it.
    private final class MaterializerFixture {

        let modelDirectoryUrl: URL
        let materializer: Qwen35MoeDiskExpertPageMaterializer
        let pagePayloadByteCount: Int

        private init(
            modelDirectoryUrl: URL,
            materializer: Qwen35MoeDiskExpertPageMaterializer,
            pagePayloadByteCount: Int
        ) {
            self.modelDirectoryUrl = modelDirectoryUrl
            self.materializer = materializer
            self.pagePayloadByteCount = pagePayloadByteCount
        }

        fileprivate static func build(
            projectionStorageByName: [String: ExpertLayerQuantizationMode],
            expertIds: [Int],
            expertFileReadMetrics: PositionalFileReadMetrics? = nil,
            writeAffineCompanionTensors: Bool = true
        ) throws -> MaterializerFixture {
            var tensorPlans: [ShardTensorPlan] = []
            for (projectionName, projectionStorage): (String, ExpertLayerQuantizationMode)
            in projectionStorageByName {
                tensorPlans.append(ShardTensorPlan(
                    tensorName: "\(projectionName).weight",
                    fillBaseByte: fillBaseByte(projectionName: projectionName)))
                if projectionStorage == .affine && writeAffineCompanionTensors {
                    tensorPlans.append(ShardTensorPlan(
                        tensorName: "\(projectionName).scales",
                        fillBaseByte: fillBaseByte(projectionName: projectionName)))
                    tensorPlans.append(ShardTensorPlan(
                        tensorName: "\(projectionName).biases",
                        fillBaseByte: fillBaseByte(projectionName: projectionName)))
                }
            }
            let modelDirectoryUrl: URL = FileManager.default.temporaryDirectory
                .appendingPathComponent("disk-materializer-\(UUID().uuidString)")
            try FileManager.default.createDirectory(
                at: modelDirectoryUrl, withIntermediateDirectories: true)
            let shardFileUrl: URL = modelDirectoryUrl
                .appendingPathComponent(Qwen35MoeDiskExpertPageMaterializerTests.SHARD_FILE_NAME)
            let writtenShard: WrittenShard = try Qwen35MoeDiskExpertPageMaterializerTests
                .writeShardFile(fileUrl: shardFileUrl, tensorPlans: tensorPlans)
            let tensorSources: [QuantizedTensorSource] = writtenShard.writtenTensors.map(
                { (writtenTensor: WrittenTensor) -> QuantizedTensorSource in
                    let nameParts: [String] = writtenTensor.tensorName.components(
                        separatedBy: ".")
                    let projectionName: String = nameParts.first ?? ""
                    let parameterName: String = nameParts.last ?? ""
                    let projectionIsAffine: Bool =
                        projectionStorageByName[projectionName] == .affine
                    return QuantizedTensorSource(
                        tensorName: writtenTensor.tensorName,
                        projectionName: projectionName,
                        parameterName: parameterName,
                        quantizationBits: projectionIsAffine ? 4 : 0,
                        quantizationGroupSize: projectionIsAffine ? 32 : 0,
                        sourceFileName: Qwen35MoeDiskExpertPageMaterializerTests.SHARD_FILE_NAME,
                        sourceFileSizeBytes: writtenShard.fileSizeBytes,
                        dtype: .u8,
                        fullShape: [
                            Qwen35MoeDiskExpertPageMaterializerTests.EXPERT_CAPACITY,
                            Qwen35MoeDiskExpertPageMaterializerTests.SLICE_ROWS,
                            Qwen35MoeDiskExpertPageMaterializerTests.SLICE_COLUMNS,
                        ],
                        tensorPayloadOffsetBytes: UInt64(writtenTensor.payloadOffsetBytes),
                        bytesPerExpert: Qwen35MoeDiskExpertPageMaterializerTests.EXPERT_PAYLOAD_BYTES,
                        expertCapacity: Qwen35MoeDiskExpertPageMaterializerTests.EXPERT_CAPACITY)
                })
            let layerMode: ExpertLayerQuantizationMode =
                projectionStorageByName.values.allSatisfy(
                    { (storage: ExpertLayerQuantizationMode) -> Bool in
                        return storage == .nativeBfloat16
                    }) ? .nativeBfloat16 : .affine
            let layerPlan: QuantizedExpertLayerPlan = QuantizedExpertLayerPlan(
                layerPrefix: Qwen35MoeDiskExpertPageMaterializerTests.LAYER_PREFIX,
                tensorSources: tensorSources,
                expertCapacity: Qwen35MoeDiskExpertPageMaterializerTests.EXPERT_CAPACITY,
                quantizationBits: layerMode == .affine ? 4 : 0,
                quantizationGroupSize: layerMode == .affine ? 32 : 0,
                quantizationMode: layerMode,
                quantizationModeByProjectionName: projectionStorageByName)
            return MaterializerFixture(
                modelDirectoryUrl: modelDirectoryUrl,
                materializer: Qwen35MoeDiskExpertPageMaterializer(
                    modelDirectory: modelDirectoryUrl,
                    layerPlans: [layerPlan],
                    expertFileReadMetrics: expertFileReadMetrics,
                    attributionEnabled: false),
                pagePayloadByteCount: expertIds.count * tensorSources.count
                    * Qwen35MoeDiskExpertPageMaterializerTests.EXPERT_PAYLOAD_BYTES)
        }

        fileprivate func cleanUp() -> Void {
            try? FileManager.default.removeItem(at: self.modelDirectoryUrl)
        }
    }

    private static func fillBaseByte(projectionName: String) -> UInt8 {
        switch projectionName {
        case "gate_proj": return 0
        case "up_proj": return 40
        case "down_proj": return 80
        default: return 120
        }
    }

    private struct ShardTensorPlan {
        let tensorName: String
        let fillBaseByte: UInt8
    }

    private struct WrittenTensor {
        let tensorName: String
        let payloadOffsetBytes: Int
    }

    private struct WrittenShard {
        let writtenTensors: [WrittenTensor]
        let fileSizeBytes: UInt64
    }

    private static func writeShardFile(
        fileUrl: URL,
        tensorPlans: [ShardTensorPlan]
    ) throws -> WrittenShard {
        var payloadBytes: Data = Data()
        var relativePayloadStartBytes: [Int] = []
        var headerEntries: [String] = []
        let expertCapacity: Int = Qwen35MoeDiskExpertPageMaterializerTests.EXPERT_CAPACITY
        for tensorPlan: ShardTensorPlan in tensorPlans {
            let relativePayloadStartByte: Int = payloadBytes.count
            relativePayloadStartBytes.append(relativePayloadStartByte)
            for expertIndex: Int in 0 ..< expertCapacity {
                payloadBytes.append(Data(
                    repeating: tensorPlan.fillBaseByte &+ UInt8(truncatingIfNeeded: expertIndex),
                    count: Qwen35MoeDiskExpertPageMaterializerTests.EXPERT_PAYLOAD_BYTES))
            }
            let relativePayloadEndByte: Int = payloadBytes.count
            // Safetensors offsets are relative to the data section, so the
            // header can be sized before absolute positions exist.
            headerEntries.append(
                "\"\(tensorPlan.tensorName)\": {\"dtype\": \"U8\", "
                    + "\"shape\": [\(expertCapacity), "
                    + "\(Qwen35MoeDiskExpertPageMaterializerTests.SLICE_ROWS), "
                    + "\(Qwen35MoeDiskExpertPageMaterializerTests.SLICE_COLUMNS)], "
                    + "\"data_offsets\": [\(relativePayloadStartByte), "
                    + "\(relativePayloadEndByte)]}")
        }
        let headerJsonText: String = "{" + headerEntries.joined(separator: ",") + "}"
        let headerBytes: Data = Data(headerJsonText.utf8)
        let dataSectionStartByte: Int = 8 + headerBytes.count
        var fileBytes: Data = Data()
        withUnsafeBytes(
            of: UInt64(headerBytes.count).littleEndian,
            { (lengthBuffer: UnsafeRawBufferPointer) -> Void in
                fileBytes.append(contentsOf: lengthBuffer)
            })
        fileBytes.append(headerBytes)
        fileBytes.append(payloadBytes)
        try fileBytes.write(to: fileUrl)
        let writtenTensors: [WrittenTensor] = tensorPlans.enumerated().map(
            { (tensorPlanIndex: Int, tensorPlan: ShardTensorPlan) -> WrittenTensor in
                return WrittenTensor(
                    tensorName: tensorPlan.tensorName,
                    payloadOffsetBytes: dataSectionStartByte
                        + relativePayloadStartBytes[tensorPlanIndex])
            })
        return WrittenShard(
            writtenTensors: writtenTensors,
            fileSizeBytes: UInt64(fileBytes.count))
    }
}

}
