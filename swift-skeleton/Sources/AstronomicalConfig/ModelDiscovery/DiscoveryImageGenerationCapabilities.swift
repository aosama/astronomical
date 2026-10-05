import Foundation;

/** Image-serving capabilities and dimension rules, in pixels and steps. */
internal struct DiscoveryImageGenerationCapabilities: Equatable, Sendable {
    internal let supportsTextToImage: Bool;
    internal let supportsImageEditing: Bool;
    internal let supportsMultipleReferenceImages: Bool;
    internal let defaultSteps: UInt16;
    internal let minimumDimensionPixels: UInt32;
    internal let maximumDimensionPixels: UInt32;
    internal let dimensionMultiplePixels: UInt32;

    internal init(
        supportsTextToImage: Bool,
        supportsImageEditing: Bool,
        supportsMultipleReferenceImages: Bool,
        defaultSteps: UInt16,
        minimumDimensionPixels: UInt32,
        maximumDimensionPixels: UInt32,
        dimensionMultiplePixels: UInt32
    ) {
        self.supportsTextToImage = supportsTextToImage;
        self.supportsImageEditing = supportsImageEditing;
        self.supportsMultipleReferenceImages = supportsMultipleReferenceImages;
        self.defaultSteps = defaultSteps;
        self.minimumDimensionPixels = minimumDimensionPixels;
        self.maximumDimensionPixels = maximumDimensionPixels;
        self.dimensionMultiplePixels = dimensionMultiplePixels;
    }

    /**
     * First violated dimension rule for a requested resolution, or nil when
     * the request fits the model's accepted dimension grid.
     */
    internal func imageDimensionViolation(
        modelId: String,
        widthPixels: UInt32,
        heightPixels: UInt32
    ) -> Optional<(parameterName: String, violationMessage: String)> {
        if let widthViolation: (parameterName: String, violationMessage: String) = self.dimensionViolation(
            modelId: modelId,
            parameterName: "width",
            dimensionPixels: widthPixels
        ) {
            return widthViolation;
        }
        return self.dimensionViolation(
            modelId: modelId,
            parameterName: "height",
            dimensionPixels: heightPixels
        );
    }

    private func dimensionViolation(
        modelId: String,
        parameterName: String,
        dimensionPixels: UInt32
    ) -> Optional<(parameterName: String, violationMessage: String)> {
        if dimensionPixels < self.minimumDimensionPixels {
            return (
                parameterName: parameterName,
                violationMessage: "\(parameterName) must be at least \(self.minimumDimensionPixels) pixels for \(modelId), received \(dimensionPixels)"
            );
        }
        if dimensionPixels > self.maximumDimensionPixels {
            return (
                parameterName: parameterName,
                violationMessage: "\(parameterName) must be at most \(self.maximumDimensionPixels) pixels for \(modelId), received \(dimensionPixels)"
            );
        }
        if self.dimensionMultiplePixels > 0 && dimensionPixels % self.dimensionMultiplePixels != 0 {
            return (
                parameterName: parameterName,
                violationMessage: "\(parameterName) must be a multiple of \(self.dimensionMultiplePixels) pixels for \(modelId), received \(dimensionPixels)"
            );
        }
        return nil;
    }
}
