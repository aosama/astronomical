import CoreGraphics;
import Foundation;
import ImageIO;

/// A bounded semantic validation failure in one image-generation command.
public enum ImageGenerationValidationError: Error, Equatable, CustomStringConvertible {
    case emptyModelId;
    case emptyPrompt;
    case widthOutOfRange(actualWidthPixels: UInt32, minimumWidthPixels: UInt32, maximumWidthPixels: UInt32);
    case widthNotAligned(actualWidthPixels: UInt32, requiredMultiplePixels: UInt32);
    case heightOutOfRange(actualHeightPixels: UInt32, minimumHeightPixels: UInt32, maximumHeightPixels: UInt32);
    case heightNotAligned(actualHeightPixels: UInt32, requiredMultiplePixels: UInt32);
    case stepsOutOfRange(actualSteps: UInt16, minimumSteps: UInt16, maximumSteps: UInt16);
    case guidanceOutOfRange(actualGuidanceThousandths: UInt32, maximumGuidanceThousandths: UInt32);

    public var description: String {
        switch (self) {
        case .emptyModelId:
            return "model ID must not be empty";
        case .emptyPrompt:
            return "image prompt must not be empty";
        case let .widthOutOfRange(actualWidthPixels, minimumWidthPixels, maximumWidthPixels):
            return "image width \(actualWidthPixels) is outside \(minimumWidthPixels)..=\(maximumWidthPixels) pixels";
        case let .widthNotAligned(actualWidthPixels, requiredMultiplePixels):
            return "image width \(actualWidthPixels) must be a multiple of \(requiredMultiplePixels) pixels";
        case let .heightOutOfRange(actualHeightPixels, minimumHeightPixels, maximumHeightPixels):
            return "image height \(actualHeightPixels) is outside \(minimumHeightPixels)..=\(maximumHeightPixels) pixels";
        case let .heightNotAligned(actualHeightPixels, requiredMultiplePixels):
            return "image height \(actualHeightPixels) must be a multiple of \(requiredMultiplePixels) pixels";
        case let .stepsOutOfRange(actualSteps, minimumSteps, maximumSteps):
            return "image step count \(actualSteps) is outside \(minimumSteps)..=\(maximumSteps)";
        case let .guidanceOutOfRange(actualGuidanceThousandths, maximumGuidanceThousandths):
            return "image guidance \(actualGuidanceThousandths) thousandths exceeds \(maximumGuidanceThousandths)";
        }
    }
}

/// A malformed image completion received from the worker process; public because
/// ProtocolError carries it as an associated payload. PngValidation throws it
/// from another file.
public enum ImageGenerationCompletionValidationError: Error, Equatable, CustomStringConvertible {
    case invalidMetadata(metadataError: ImageGenerationValidationError);
    case invalidMimeType;
    case invalidPngEncoding;
    case nonRgb8Png;
    case pngDecodeResourceLimit;
    case pngDimensionsMismatch(
        encodedWidthPixels: UInt32,
        encodedHeightPixels: UInt32,
        metadataWidthPixels: UInt32,
        metadataHeightPixels: UInt32);

    public var description: String {
        switch (self) {
        case .invalidMetadata:
            return "image completion metadata is invalid";
        case .invalidMimeType:
            return "completed image MIME type must be exactly image/png";
        case .invalidPngEncoding:
            return "completed image is not a fully decodable PNG";
        case .nonRgb8Png:
            return "completed PNG must use lossless 8-bit RGB pixels";
        case .pngDecodeResourceLimit:
            return "completed PNG exceeds the bounded decode resource limit";
        case let .pngDimensionsMismatch(encodedWidthPixels, encodedHeightPixels, metadataWidthPixels, metadataHeightPixels):
            return "completed PNG dimensions \(encodedWidthPixels)x\(encodedHeightPixels) do not match metadata \(metadataWidthPixels)x\(metadataHeightPixels)";
        }
    }
}

/// A bounded semantic failure in a worker capability advertisement; public
/// because ProtocolError carries it as an associated payload.
public enum WorkerModelCapabilitiesValidationError: Error, Equatable, CustomStringConvertible {
    case noCapabilities;
    case zeroEmbeddingVectorWidth;
    case embeddingContextTooSmall(actualContextTokens: UInt32);
    case zeroImageDimensionAlignment;
    case invertedImageWidthBounds(minimumWidthPixels: UInt32, maximumWidthPixels: UInt32);
    case invertedImageHeightBounds(minimumHeightPixels: UInt32, maximumHeightPixels: UInt32);
    case emptyImageOutputMimeType;

    public var description: String {
        switch (self) {
        case .noCapabilities:
            return "worker model must advertise at least one capability";
        case .zeroEmbeddingVectorWidth:
            return "embedding vector width must be positive";
        case let .embeddingContextTooSmall(actualContextTokens):
            return "embedding context must hold at least two positions, got \(actualContextTokens)";
        case .zeroImageDimensionAlignment:
            return "image dimension alignment must be positive";
        case let .invertedImageWidthBounds(minimumWidthPixels, maximumWidthPixels):
            return "image width bounds are inverted: minimum \(minimumWidthPixels) exceeds maximum \(maximumWidthPixels)";
        case let .invertedImageHeightBounds(minimumHeightPixels, maximumHeightPixels):
            return "image height bounds are inverted: minimum \(minimumHeightPixels) exceeds maximum \(maximumHeightPixels)";
        case .emptyImageOutputMimeType:
            return "image output MIME types must contain only nonempty values";
        }
    }
}

private enum ImageGenerationValidation {
    static let minimumImageDimensionPixels: UInt32 = 64;
    static let maximumImageDimensionPixels: UInt32 = 16_384;
    static let imageDimensionMultiplePixels: UInt32 = 8;
    static let minimumImageGenerationSteps: UInt16 = 1;
    static let maximumImageGenerationSteps: UInt16 = 1_000;
    static let maximumImageGuidanceThousandths: UInt32 = 100_000;
    // A completion must not expand beyond the transport budget into a much larger memory owner.
    static let maximumDecodedRgbBytes: Int = IpcFrameLimits.maximumIpcFrameBytes;

    static func validateCommand(_ command: ImageGenerationCommand) throws {
        if command.model.trimmingCharacters(in: CharacterSet.whitespacesAndNewlines).isEmpty {
            throw ImageGenerationValidationError.emptyModelId;
        }
        if command.prompt.trimmingCharacters(in: CharacterSet.whitespacesAndNewlines).isEmpty {
            throw ImageGenerationValidationError.emptyPrompt;
        }
        try ImageGenerationValidation.validateSettings(command.settings);
    }

    /// Enforces protocol-wide safety bounds before model-specific capability checks.
    static func validateSettings(_ settings: ImageGenerationSettings) throws {
        if settings.widthPixels < ImageGenerationValidation.minimumImageDimensionPixels
            || settings.widthPixels > ImageGenerationValidation.maximumImageDimensionPixels {
            throw ImageGenerationValidationError.widthOutOfRange(
                actualWidthPixels: settings.widthPixels,
                minimumWidthPixels: ImageGenerationValidation.minimumImageDimensionPixels,
                maximumWidthPixels: ImageGenerationValidation.maximumImageDimensionPixels);
        }
        if settings.widthPixels % ImageGenerationValidation.imageDimensionMultiplePixels != 0 {
            throw ImageGenerationValidationError.widthNotAligned(
                actualWidthPixels: settings.widthPixels,
                requiredMultiplePixels: ImageGenerationValidation.imageDimensionMultiplePixels);
        }
        if settings.heightPixels < ImageGenerationValidation.minimumImageDimensionPixels
            || settings.heightPixels > ImageGenerationValidation.maximumImageDimensionPixels {
            throw ImageGenerationValidationError.heightOutOfRange(
                actualHeightPixels: settings.heightPixels,
                minimumHeightPixels: ImageGenerationValidation.minimumImageDimensionPixels,
                maximumHeightPixels: ImageGenerationValidation.maximumImageDimensionPixels);
        }
        if settings.heightPixels % ImageGenerationValidation.imageDimensionMultiplePixels != 0 {
            throw ImageGenerationValidationError.heightNotAligned(
                actualHeightPixels: settings.heightPixels,
                requiredMultiplePixels: ImageGenerationValidation.imageDimensionMultiplePixels);
        }
        if settings.steps < ImageGenerationValidation.minimumImageGenerationSteps
            || settings.steps > ImageGenerationValidation.maximumImageGenerationSteps {
            throw ImageGenerationValidationError.stepsOutOfRange(
                actualSteps: settings.steps,
                minimumSteps: ImageGenerationValidation.minimumImageGenerationSteps,
                maximumSteps: ImageGenerationValidation.maximumImageGenerationSteps);
        }
        if settings.guidanceThousandths > ImageGenerationValidation.maximumImageGuidanceThousandths {
            throw ImageGenerationValidationError.guidanceOutOfRange(
                actualGuidanceThousandths: settings.guidanceThousandths,
                maximumGuidanceThousandths: ImageGenerationValidation.maximumImageGuidanceThousandths);
        }
    }

    /// Completion facts must obey the same protocol bounds as the accepted request.
    static func validateResultMetadata(_ metadata: ImageGenerationResultMetadata) throws {
        let equivalentSettings: ImageGenerationSettings = ImageGenerationSettings(
            widthPixels: metadata.widthPixels,
            heightPixels: metadata.heightPixels,
            steps: metadata.steps,
            guidanceThousandths: metadata.guidanceThousandths,
            seed: metadata.seed);
        try ImageGenerationValidation.validateSettings(equivalentSettings);
    }

    /// Rejects capability advertisements that cannot safely constrain a request.
    static func validateImageCapabilities(_ capabilities: ImageGenerationCapabilities) throws {
        if capabilities.dimensionMultiplePixels == 0 {
            throw WorkerModelCapabilitiesValidationError.zeroImageDimensionAlignment;
        }
        if capabilities.minimumWidthPixels > capabilities.maximumWidthPixels {
            throw WorkerModelCapabilitiesValidationError.invertedImageWidthBounds(
                minimumWidthPixels: capabilities.minimumWidthPixels,
                maximumWidthPixels: capabilities.maximumWidthPixels);
        }
        if capabilities.minimumHeightPixels > capabilities.maximumHeightPixels {
            throw WorkerModelCapabilitiesValidationError.invertedImageHeightBounds(
                minimumHeightPixels: capabilities.minimumHeightPixels,
                maximumHeightPixels: capabilities.maximumHeightPixels);
        }
        if capabilities.outputMimeTypes.isEmpty
            || ImageGenerationValidation.hasEmptyOutputMimeType(capabilities.outputMimeTypes) {
            throw WorkerModelCapabilitiesValidationError.emptyImageOutputMimeType;
        }
    }

    /// Enforces positive geometry before the supervisor can advertise the endpoint.
    static func validateEmbeddingCapabilities(_ capabilities: WorkerEmbeddingCapabilities) throws {
        if capabilities.vectorWidth == 0 {
            throw WorkerModelCapabilitiesValidationError.zeroEmbeddingVectorWidth;
        }
        if capabilities.maxInputTokens < 2 {
            throw WorkerModelCapabilitiesValidationError.embeddingContextTooSmall(
                actualContextTokens: capabilities.maxInputTokens);
        }
    }

    /// A loaded model must advertise at least one usable operation surface.
    static func validateWorkerModelCapabilities(_ capabilities: WorkerModelCapabilities) throws {
        if capabilities.chat == nil && capabilities.imageGeneration == nil && capabilities.embeddings == nil {
            throw WorkerModelCapabilitiesValidationError.noCapabilities;
        }
        if let imageCapabilities = capabilities.imageGeneration {
            try ImageGenerationValidation.validateImageCapabilities(imageCapabilities);
        }
        if let embeddingCapabilities = capabilities.embeddings {
            try ImageGenerationValidation.validateEmbeddingCapabilities(embeddingCapabilities);
        }
    }

    /// Fully decodes the PNG before trusting worker completion metadata.
    static func validateCompletion(image: GeneratedImage, resultMetadata: ImageGenerationResultMetadata) throws {
        do {
            try ImageGenerationValidation.validateResultMetadata(resultMetadata);
        } catch let metadataError as ImageGenerationValidationError {
            throw ImageGenerationCompletionValidationError.invalidMetadata(metadataError: metadataError);
        }
        if image.mimeType != PngValidation.pngMimeType {
            throw ImageGenerationCompletionValidationError.invalidMimeType;
        }
        let validatedPng: ValidatedPng = try PngValidation.validatePngStructure(
            pngBytes: image.encodedBytes,
            maximumDecodedRgbBytes: ImageGenerationValidation.maximumDecodedRgbBytes);
        if validatedPng.widthPixels != resultMetadata.widthPixels
            || validatedPng.heightPixels != resultMetadata.heightPixels {
            throw ImageGenerationCompletionValidationError.pngDimensionsMismatch(
                encodedWidthPixels: validatedPng.widthPixels,
                encodedHeightPixels: validatedPng.heightPixels,
                metadataWidthPixels: resultMetadata.widthPixels,
                metadataHeightPixels: resultMetadata.heightPixels);
        }
        if validatedPng.widthPixels > ImageGenerationValidation.maximumImageDimensionPixels
            || validatedPng.heightPixels > ImageGenerationValidation.maximumImageDimensionPixels {
            throw ImageGenerationCompletionValidationError.pngDecodeResourceLimit;
        }
        let decodeOptions: Dictionary<CFString, Bool> = [kCGImageSourceShouldCacheImmediately: true];
        guard let pngSource: CGImageSource = CGImageSourceCreateWithData(
            Data(image.encodedBytes) as CFData, decodeOptions as CFDictionary) else {
            throw ImageGenerationCompletionValidationError.invalidPngEncoding;
        }
        if CGImageSourceGetCount(pngSource) < 1 {
            throw ImageGenerationCompletionValidationError.invalidPngEncoding;
        }
        guard let decodedImage: CGImage = CGImageSourceCreateImageAtIndex(pngSource, 0, decodeOptions as CFDictionary) else {
            throw ImageGenerationCompletionValidationError.invalidPngEncoding;
        }
        if decodedImage.width != Int(validatedPng.widthPixels)
            || decodedImage.height != Int(validatedPng.heightPixels) {
            throw ImageGenerationCompletionValidationError.invalidPngEncoding;
        }
        var decodedRgbPixels: Data = Data(count: validatedPng.decodedRgbByteCount);
        let renderSucceeded: Bool = decodedRgbPixels.withUnsafeMutableBytes({ (mutableRawBuffer: UnsafeMutableRawBufferPointer) -> Bool in
            return ImageGenerationValidation.renderDecodedImage(
                decodedImage,
                into: mutableRawBuffer,
                widthPixels: validatedPng.widthPixels,
                heightPixels: validatedPng.heightPixels);
        });
        if renderSucceeded == false {
            throw ImageGenerationCompletionValidationError.invalidPngEncoding;
        }
    }

    /// Rendering through an sRGB 8-bit bitmap context forces a full pixel decode,
    /// matching the Rust decoder's read of the complete RGB8 image.
    private static func renderDecodedImage(
        _ decodedImage: CGImage,
        into pixelBuffer: UnsafeMutableRawBufferPointer,
        widthPixels: UInt32,
        heightPixels: UInt32
    ) -> Bool {
        guard let pixelBaseAddress: UnsafeMutableRawPointer = pixelBuffer.baseAddress else {
            return false;
        }
        guard let sRgbColorSpace: CGColorSpace = CGColorSpace(name: CGColorSpace.sRGB) else {
            return false;
        }
        // Quartz has no supported 24-bpp no-alpha RGB context, so the decode
        // goes through a 32-bpp skipped-alpha buffer that is compacted to RGB8.
        let pixelCount: Int = Int(widthPixels) * Int(heightPixels);
        var skippedAlphaPixels: Array<UInt8> = Array<UInt8>(repeating: 0, count: pixelCount * 4);
        guard let bitmapContext: CGContext = CGContext(
            data: &skippedAlphaPixels,
            width: Int(widthPixels),
            height: Int(heightPixels),
            bitsPerComponent: 8,
            bytesPerRow: Int(widthPixels) * 4,
            space: sRgbColorSpace,
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else {
            return false;
        }
        bitmapContext.draw(decodedImage, in: CGRect(x: 0, y: 0, width: CGFloat(widthPixels), height: CGFloat(heightPixels)));
        let rgbBaseAddress: UnsafeMutablePointer<UInt8> = pixelBaseAddress.assumingMemoryBound(to: UInt8.self);
        skippedAlphaPixels.withUnsafeBufferPointer({ (skippedAlphaBuffer: UnsafeBufferPointer<UInt8>) -> Void in
            for pixelIndex in 0..<pixelCount {
                let skippedAlphaStart: Int = pixelIndex * 4;
                let rgbStart: Int = pixelIndex * 3;
                rgbBaseAddress[rgbStart] = skippedAlphaBuffer[skippedAlphaStart];
                rgbBaseAddress[rgbStart + 1] = skippedAlphaBuffer[skippedAlphaStart + 1];
                rgbBaseAddress[rgbStart + 2] = skippedAlphaBuffer[skippedAlphaStart + 2];
            }
        });
        return true;
    }

    private static func hasEmptyOutputMimeType(_ outputMimeTypes: Array<String>) -> Bool {
        for outputMimeType in outputMimeTypes {
            if outputMimeType.trimmingCharacters(in: CharacterSet.whitespacesAndNewlines).isEmpty {
                return true;
            }
        }
        return false;
    }
}

extension ImageGenerationCommand {

    /// Independently validates image input after it crosses the worker trust boundary.
    public func validate() throws {
        return try ImageGenerationValidation.validateCommand(self);
    }
}

extension ImageGenerationSettings {

    /// Enforces protocol-wide safety bounds before model-specific capability checks.
    public func validate() throws {
        return try ImageGenerationValidation.validateSettings(self);
    }
}

extension ImageGenerationResultMetadata {

    /// Completion facts must obey the same protocol bounds as the accepted request.
    public func validate() throws {
        return try ImageGenerationValidation.validateResultMetadata(self);
    }
}

extension GeneratedImage {

    /// Fully decodes the PNG before trusting worker completion metadata.
    public func validateCompletion(resultMetadata: ImageGenerationResultMetadata) throws {
        return try ImageGenerationValidation.validateCompletion(image: self, resultMetadata: resultMetadata);
    }
}

extension ImageGenerationCapabilities {

    /// Rejects capability advertisements that cannot safely constrain a request.
    public func validate() throws {
        return try ImageGenerationValidation.validateImageCapabilities(self);
    }
}

extension WorkerEmbeddingCapabilities {

    /// Enforces positive geometry before the supervisor can advertise the endpoint.
    public func validate() throws {
        return try ImageGenerationValidation.validateEmbeddingCapabilities(self);
    }
}

extension WorkerModelCapabilities {

    /// A loaded model must advertise at least one usable operation surface.
    public func validate() throws {
        return try ImageGenerationValidation.validateWorkerModelCapabilities(self);
    }
}
