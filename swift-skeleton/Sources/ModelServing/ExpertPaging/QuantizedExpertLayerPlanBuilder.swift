import IpcProtocol
import Foundation

/// Builds startup-validated expert tensor geometry from shard headers and
/// module quantization profiles.
public enum QuantizedExpertLayerPlanBuilder {

    private static let PROJECTION_NAMES: [String] = ["gate_proj", "up_proj", "down_proj"]
    private static let AFFINE_PARAMETER_NAMES: [String] = ["weight", "scales", "biases"]
    private static let MAXIMUM_HEADER_LENGTH_BYTES: UInt64 =
        BoundedSafetensors.MAXIMUM_ARTIFACT_SAFETENSORS_HEADER_LENGTH_BYTES

    private struct ParsedShardHeader {
        let dataSectionStartBytes: UInt64
        let fileSizeBytes: UInt64
        let tensorViewsByName: [String: SafetensorsFraming.TensorView]
    }

    private struct ProjectionStorage {
        let bits: Int32
        let groupSize: Int32
        let parameterNames: [String]
        let mode: ExpertLayerQuantizationMode
    }

    /// Builds one validated layer plan and parses each referenced shard header once.
    public static func buildLayerPlan(
        modelDirectory: URL,
        weightMap: [String: String],
        layerPrefix: String,
        config: Qwen3_5Config
    ) throws -> QuantizedExpertLayerPlan {
        var headerCache: [URL: ParsedShardHeader] = [:]
        return try QuantizedExpertLayerPlanBuilder.buildLayerPlan(
            modelDirectory: modelDirectory,
            weightMap: weightMap,
            layerPrefix: layerPrefix,
            config: config,
            headerCache: &headerCache)
    }

    /// Builds every decoder layer's plan over one shared shard-header cache,
    /// so a multi-layer model parses each shard header exactly once instead
    /// of once per layer.
    public static func buildLayerPlans(
        modelDirectory: URL,
        weightMap: [String: String],
        layerPrefixes: [String],
        config: Qwen3_5Config
    ) throws -> [QuantizedExpertLayerPlan] {
        var headerCache: [URL: ParsedShardHeader] = [:]
        return try layerPrefixes.map({ (layerPrefix: String) -> QuantizedExpertLayerPlan in
            return try QuantizedExpertLayerPlanBuilder.buildLayerPlan(
                modelDirectory: modelDirectory,
                weightMap: weightMap,
                layerPrefix: layerPrefix,
                config: config,
                headerCache: &headerCache)
        })
    }

    private static func buildLayerPlan(
        modelDirectory: URL,
        weightMap: [String: String],
        layerPrefix: String,
        config: Qwen3_5Config,
        headerCache: inout [URL: ParsedShardHeader]
    ) throws -> QuantizedExpertLayerPlan {
        var tensorSources: [QuantizedTensorSource] = []
        var modeByProjectionName: [String: ExpertLayerQuantizationMode] = [:]
        for projectionName: String in QuantizedExpertLayerPlanBuilder.PROJECTION_NAMES {
            let projectionModuleName: String = "\(layerPrefix).switch_mlp.\(projectionName)"
            let profile: OptiQQuantizationProfile = config.quantizationProfile(forModule: projectionModuleName)
            let storage: ProjectionStorage = try QuantizedExpertLayerPlanBuilder.storageContract(profile: profile)
            modeByProjectionName[projectionName] = storage.mode
            var sourcesByParameterName: [String: QuantizedTensorSource] = [:]
            for parameterName: String in storage.parameterNames {
                let canonicalTensorName: String = "\(projectionModuleName).\(parameterName)"
                guard let sourceFileName: String = weightMap[canonicalTensorName] else {
                    throw ExpertPagingError.manifestValidationFailure(
                        description: "missing shard-index entry for tensor \(canonicalTensorName)")
                }
                guard URL(fileURLWithPath: sourceFileName).lastPathComponent == sourceFileName,
                    sourceFileName.isEmpty == false else {
                    throw ExpertPagingError.manifestValidationFailure(
                        description: "shard filename must be a single file name: \(sourceFileName)")
                }
                let sourceFileUrl: URL = modelDirectory.appendingPathComponent(sourceFileName)
                let parsedHeader: ParsedShardHeader
                if let cachedHeader: ParsedShardHeader = headerCache[sourceFileUrl] {
                    parsedHeader = cachedHeader
                } else {
                    let loadedHeader: ParsedShardHeader = try QuantizedExpertLayerPlanBuilder
                        .readShardHeader(sourceFileUrl: sourceFileUrl)
                    headerCache[sourceFileUrl] = loadedHeader
                    parsedHeader = loadedHeader
                }
                guard let tensorView: SafetensorsFraming.TensorView =
                    parsedHeader.tensorViewsByName[canonicalTensorName] else {
                    throw ExpertPagingError.manifestValidationFailure(
                        description: "shard \(sourceFileName) does not contain tensor \(canonicalTensorName)")
                }
                guard let tensorDtype: SafetensorsDtype =
                    SafetensorsDtype.parsed(fromCanonicalName: tensorView.dtype) else {
                    throw ExpertPagingError.manifestValidationFailure(
                        description: "tensor \(canonicalTensorName) has unsupported dtype \(tensorView.dtype)")
                }
                let source: QuantizedTensorSource = try QuantizedExpertLayerPlanBuilder
                    .validatedTensorSource(
                        tensorName: canonicalTensorName,
                        projectionName: projectionName,
                        parameterName: parameterName,
                        storage: storage,
                        tensorView: tensorView,
                        dtype: tensorDtype,
                        sourceFileName: sourceFileName,
                        sourceFileSizeBytes: parsedHeader.fileSizeBytes,
                        dataSectionStartBytes: parsedHeader.dataSectionStartBytes)
                sourcesByParameterName[parameterName] = source
            }
            if storage.mode == .affine {
                guard let weightSource: QuantizedTensorSource = sourcesByParameterName["weight"],
                    let scalesSource: QuantizedTensorSource = sourcesByParameterName["scales"],
                    let biasesSource: QuantizedTensorSource = sourcesByParameterName["biases"] else {
                    throw ExpertPagingError.manifestValidationFailure(
                        description: "affine projection \(projectionName) is missing a companion tensor")
                }
                try QuantizedExpertLayerPlanBuilder.validateProjectionGeometry(
                    projectionName: projectionName,
                    weightSource: weightSource,
                    scalesSource: scalesSource,
                    biasesSource: biasesSource,
                    bits: storage.bits,
                    groupSize: storage.groupSize)
            }
            for parameterName: String in storage.parameterNames {
                guard let tensorSource: QuantizedTensorSource = sourcesByParameterName[parameterName] else {
                    throw ExpertPagingError.manifestValidationFailure(
                        description: "missing tensor source for \(projectionName).\(parameterName)")
                }
                tensorSources.append(tensorSource)
            }
        }
        let expertCapacities: Set<Int> = Set(tensorSources.map({ (source: QuantizedTensorSource) -> Int in
            return source.expertCapacity
        }))
        guard expertCapacities.count == 1, let expertCapacity: Int = expertCapacities.first else {
            throw ExpertPagingError.manifestValidationFailure(
                description: "expert tensors in layer \(layerPrefix) have inconsistent capacities")
        }
        let layerMode: ExpertLayerQuantizationMode = modeByProjectionName.values.allSatisfy({
            (mode: ExpertLayerQuantizationMode) -> Bool in mode == .nativeBfloat16
        }) ? .nativeBfloat16 : .affine
        let layerBits: Int32 = layerMode == .affine
            ? try QuantizedExpertLayerPlanBuilder.checkedInt32(config.defaultQuantizationBits())
            : 0
        let layerGroupSize: Int32 = layerMode == .affine
            ? try QuantizedExpertLayerPlanBuilder.checkedInt32(config.defaultQuantizationGroupSize())
            : 0
        return QuantizedExpertLayerPlan(
            layerPrefix: layerPrefix,
            tensorSources: tensorSources,
            expertCapacity: expertCapacity,
            quantizationBits: layerBits,
            quantizationGroupSize: layerGroupSize,
            quantizationMode: layerMode,
            quantizationModeByProjectionName: modeByProjectionName)
    }

    private static func readShardHeader(sourceFileUrl: URL) throws -> ParsedShardHeader {
        let fileHandle: FileHandle
        do {
            fileHandle = try FileHandle(forReadingFrom: sourceFileUrl)
        } catch {
            throw ExpertPagingError.manifestValidationFailure(
                description: "could not open shard \(sourceFileUrl.lastPathComponent): \(error)")
        }
        defer { try? fileHandle.close() }
        let fileSizeBytes: UInt64
        do {
            let fileAttributes: [FileAttributeKey: Any] = try FileManager.default.attributesOfItem(
                atPath: sourceFileUrl.path)
            guard let fileSizeNumber: NSNumber = fileAttributes[.size] as? NSNumber,
                fileSizeNumber.int64Value >= 0 else {
                throw ExpertPagingError.manifestValidationFailure(
                    description: "could not determine shard size for \(sourceFileUrl.lastPathComponent)")
            }
            fileSizeBytes = fileSizeNumber.uint64Value
        } catch let pagingError as ExpertPagingError {
            throw pagingError
        } catch {
            throw ExpertPagingError.manifestValidationFailure(
                description: "could not inspect shard \(sourceFileUrl.lastPathComponent): \(error)")
        }
        let boundedHeader: SafetensorsFraming.BoundedJsonHeader
        do {
            boundedHeader = try SafetensorsFraming.readBoundedJsonHeader(
                fileHandle: fileHandle,
                fileSizeBytes: fileSizeBytes,
                maximumHeaderLengthBytes: QuantizedExpertLayerPlanBuilder.MAXIMUM_HEADER_LENGTH_BYTES)
        } catch {
            throw ExpertPagingError.manifestValidationFailure(
                description: "could not parse shard header \(sourceFileUrl.lastPathComponent): \(error)")
        }
        var tensorViewsByName: [String: SafetensorsFraming.TensorView] = [:]
        tensorViewsByName.reserveCapacity(boundedHeader.tensorJsonValues.count)
        for headerEntry: (tensorName: String, headerValue: JsonWireValue) in boundedHeader.tensorJsonValues {
            do {
                tensorViewsByName[headerEntry.tensorName] = try SafetensorsFraming.TensorView
                    .decoded(wireValue: headerEntry.headerValue)
            } catch let jsonWireProblem as JsonWireProblem {
                throw ExpertPagingError.manifestValidationFailure(
                    description: "invalid tensor declaration \(headerEntry.tensorName) in shard "
                        + "\(sourceFileUrl.lastPathComponent): \(jsonWireProblem.description)")
            }
        }
        return ParsedShardHeader(
            dataSectionStartBytes: boundedHeader.dataSectionStartBytes,
            fileSizeBytes: fileSizeBytes,
            tensorViewsByName: tensorViewsByName)
    }

    private static func storageContract(profile: OptiQQuantizationProfile) throws -> ProjectionStorage {
        if profile.isUnquantized() {
            return ProjectionStorage(
                bits: 0,
                groupSize: 0,
                parameterNames: ["weight"],
                mode: .nativeBfloat16)
        }
        let bits: Int32 = try QuantizedExpertLayerPlanBuilder.checkedInt32(profile.bits)
        let groupSize: Int32 = try QuantizedExpertLayerPlanBuilder.checkedInt32(profile.groupSize)
        try QuantizedExpertManifestValidation.validatedQuantizationContract(
            quantizationBits: bits,
            quantizationGroupSize: groupSize)
        return ProjectionStorage(
            bits: bits,
            groupSize: groupSize,
            parameterNames: QuantizedExpertLayerPlanBuilder.AFFINE_PARAMETER_NAMES,
            mode: .affine)
    }

    private static func checkedInt32(_ unsignedValue: UInt32) throws -> Int32 {
        guard unsignedValue <= UInt32(Int32.max) else {
            throw ExpertPagingError.manifestValidationFailure(
                description: "quantization value exceeds signed 32-bit range: \(unsignedValue)")
        }
        return Int32(unsignedValue)
    }

    private static func validatedTensorSource(
        tensorName: String,
        projectionName: String,
        parameterName: String,
        storage: ProjectionStorage,
        tensorView: SafetensorsFraming.TensorView,
        dtype: SafetensorsDtype,
        sourceFileName: String,
        sourceFileSizeBytes: UInt64,
        dataSectionStartBytes: UInt64
    ) throws -> QuantizedTensorSource {
        let expectedDtype: SafetensorsDtype?
        if storage.mode == .nativeBfloat16 {
            expectedDtype = .bf16
        } else if parameterName == "weight" {
            expectedDtype = .u32
        } else {
            expectedDtype = nil
        }
        if let expectedDtype: SafetensorsDtype = expectedDtype, dtype != expectedDtype {
            throw ExpertPagingError.manifestValidationFailure(
                description: "tensor \(tensorName) must use \(expectedDtype.canonicalName), "
                    + "found \(dtype.canonicalName)")
        }
        if expectedDtype == nil,
            dtype != .f16 && dtype != .bf16 && dtype != .f32 {
            throw ExpertPagingError.manifestValidationFailure(
                description: "affine tensor \(tensorName) has unsupported dtype \(dtype.canonicalName)")
        }
        guard tensorView.shape.count == 3,
            tensorView.shape.allSatisfy({ (dimension: Int) -> Bool in dimension > 0 }) else {
            throw ExpertPagingError.manifestValidationFailure(
                description: "tensor \(tensorName) must have three non-zero dimensions")
        }
        guard tensorView.dataOffsets.count == 2,
            tensorView.dataOffsets[1] >= tensorView.dataOffsets[0] else {
            throw ExpertPagingError.manifestValidationFailure(
                description: "tensor \(tensorName) has invalid payload offsets")
        }
        let (relativeTensorByteCount, intervalUnderflowed): (UInt64, Bool) = tensorView.dataOffsets[1]
            .subtractingReportingOverflow(tensorView.dataOffsets[0])
        let expectedTensorByteCount: UInt64 = tensorView.shape.reduce(UInt64(1), { (product: UInt64, dimension: Int) -> UInt64 in
            return product * UInt64(dimension)
        }) * dtype.bitsize / 8
        guard intervalUnderflowed == false, relativeTensorByteCount == expectedTensorByteCount else {
            throw ExpertPagingError.manifestValidationFailure(
                description: "tensor \(tensorName) payload length does not match its shape and dtype")
        }
        let (absolutePayloadOffsetBytes, offsetOverflowed): (UInt64, Bool) = dataSectionStartBytes
            .addingReportingOverflow(tensorView.dataOffsets[0])
        let (absolutePayloadEndBytes, endOffsetOverflowed): (UInt64, Bool) = dataSectionStartBytes
            .addingReportingOverflow(tensorView.dataOffsets[1])
        guard offsetOverflowed == false, endOffsetOverflowed == false,
            absolutePayloadEndBytes <= sourceFileSizeBytes else {
            throw ExpertPagingError.manifestValidationFailure(
                description: "tensor \(tensorName) payload offset overflows")
        }
        let perExpertElements: Int = try QuantizedExpertLayerPlanBuilder.checkedProduct(
            tensorView.shape[1], tensorView.shape[2], tensorName: tensorName)
        let bytesPerElement: Int = Int(dtype.bitsize / 8)
        let bytesPerExpert: Int = try QuantizedExpertLayerPlanBuilder.checkedProduct(
            perExpertElements, bytesPerElement, tensorName: tensorName)
        return QuantizedTensorSource(
            tensorName: tensorName,
            projectionName: projectionName,
            parameterName: parameterName,
            quantizationBits: storage.bits,
            quantizationGroupSize: storage.groupSize,
            sourceFileName: sourceFileName,
            sourceFileSizeBytes: sourceFileSizeBytes,
            dtype: dtype,
            fullShape: tensorView.shape,
            tensorPayloadOffsetBytes: absolutePayloadOffsetBytes,
            bytesPerExpert: bytesPerExpert,
            expertCapacity: tensorView.shape[0])
    }

    private static func checkedProduct(_ leftFactor: Int, _ rightFactor: Int, tensorName: String) throws -> Int {
        let (product, overflowed): (Int, Bool) = leftFactor.multipliedReportingOverflow(by: rightFactor)
        guard overflowed == false else {
            throw ExpertPagingError.manifestValidationFailure(
                description: "tensor \(tensorName) byte geometry overflows")
        }
        return product
    }

    private static func validateProjectionGeometry(
        projectionName: String,
        weightSource: QuantizedTensorSource,
        scalesSource: QuantizedTensorSource,
        biasesSource: QuantizedTensorSource,
        bits: Int32,
        groupSize: Int32
    ) throws {
        guard scalesSource.fullShape == biasesSource.fullShape else {
            throw ExpertPagingError.manifestValidationFailure(
                description: "scales and biases shapes differ for projection \(projectionName)")
        }
        guard Array(weightSource.fullShape.prefix(2)) == Array(scalesSource.fullShape.prefix(2)) else {
            throw ExpertPagingError.manifestValidationFailure(
                description: "weight and scales batch dimensions differ for projection \(projectionName)")
        }
        let scalesWidth: Int = scalesSource.fullShape[2]
        let expectedPackedWidth: Int = try QuantizedExpertLayerPlanBuilder.checkedProduct(
            scalesWidth,
            try QuantizedExpertLayerPlanBuilder.checkedProduct(Int(groupSize), Int(bits),
                tensorName: "\(projectionName).scales") / 32,
            tensorName: "\(projectionName).weight")
        guard weightSource.fullShape[2] == expectedPackedWidth else {
            throw ExpertPagingError.manifestValidationFailure(
                description: "packed width for projection \(projectionName) must be "
                    + "\(expectedPackedWidth), found \(weightSource.fullShape[2])")
        }
    }
}
